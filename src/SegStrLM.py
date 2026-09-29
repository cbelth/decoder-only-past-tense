import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning as L

class SegStrLM(L.LightningModule):
    """
    an autoregressive transformer over segment sequences, trained to predict
    the next segment.

    attention is causal, so position t sees only segments up to t. that is what
    makes the summed token log-probabilities a real probability, and it is why
    the last position -- the <eos> a SegVocab encoding ends with -- is the only
    one that has seen the whole form, and the obvious place for a classifier
    head to pool. `pooling='mean'` averages every real position instead; see
    `mean_state` for when that turns out to matter.

    `hidden` is exposed separately from `forward` so a classifier can reuse the
    representation without reaching into the stack.
    """

    def __init__(self, vocab_size, pad_id, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, max_len=64,
                 train_ds=None, val_ds=None, batch_size=32, num_workers=2,
                 lr=1e-3, pooling='eos'):
        """
        builds the embeddings, the causal encoder stack and the LM head.
        `max_len` bounds the learned positional table and must cover the longest
        form plus its two brackets; SegStrDataset.max_len reports that.
        `train_ds`/`val_ds` are optional: supply them to let the module serve
        its own loaders, or hand loaders to Trainer.fit directly
        """
        super().__init__()
        # datasets are data, not hparams, so keep them out of the checkpoint
        self.save_hyperparameters(ignore=['train_ds', 'val_ds'])
        self.pad_id = pad_id
        self.max_len = max_len
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.seg_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        # norm_first rules out the nested-tensor fast path, so don't ask for it
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # True marks a position that may not be attended to, matching the
        # padding mask's convention; mixing a float mask with a bool one is
        # deprecated. not persistent, since it is derived from max_len
        self.register_buffer('causal',
                             torch.ones(max_len, max_len, dtype=torch.bool).triu(1),
                             persistent=False)

    def embed(self, srcs):
        """
        sums segment and position embeddings for a (batch, seq) block of ids
        """
        pos = torch.arange(srcs.size(1), device=srcs.device)
        return self.drop(self.seg_emb(srcs) + self.pos_emb(pos))

    def hidden(self, srcs, pad_mask=None):
        """
        the (batch, seq, d_model) states. `pad_mask` is the True-marks-padding
        mask from SegStrDataset.collate, recomputed if omitted
        """
        if srcs.size(1) > self.max_len:
            raise ValueError(f'sequence of {srcs.size(1)} exceeds max_len {self.max_len}')
        if pad_mask is None:
            pad_mask = srcs.eq(self.pad_id)

        size = srcs.size(1)
        return self.encoder(self.embed(srcs), mask=self.causal[:size, :size],
                            src_key_padding_mask=pad_mask, is_causal=True)

    def forward(self, srcs, pad_mask=None):
        """
        next-segment logits, (batch, seq, vocab). the logits at position t
        score the segment at t+1, so the last column predicts past the end
        """
        return self.lm_head(self.hidden(srcs, pad_mask))

    def eos_state(self, srcs, pad_mask=None):
        """
        the hidden state at <eos>, (batch, d_model) -- the pooled representation
        of the whole form, and what a classifier head reads.

        <eos> sits at the last unpadded position, which differs per row, so
        this has to gather rather than take hidden[:, -1]: that would read
        padding for every form shorter than the longest in its batch
        """
        if pad_mask is None:
            pad_mask = srcs.eq(self.pad_id)
        states = self.hidden(srcs, pad_mask)
        last = (~pad_mask).sum(1) - 1
        return states[torch.arange(states.size(0), device=states.device), last]

    def mean_state(self, srcs, pad_mask=None):
        """
        the mean of the hidden states over every real position, (batch,
        d_model), padding excluded from both the sum and the divisor.

        under causal attention this is not the usual encoder mean pool: the
        state at position t has seen only the first t segments, so this averages
        the form's PREFIXES rather than its segments read in context. that is
        why it is worth having. the <eos> state is the only one that has seen
        the whole form, but it is also a single vector trained for nothing but
        next-segment prediction, whereas the mean hands a probe the whole
        trajectory. measured across all five studies that read a pooled state,
        this lifts the FROZEN PROBE every time (+0.009 to +0.053 AUC) and
        sharply steadies it across seeds, while leaving the fine-tuned
        classifier unmoved -- once the label gradient reaches the backbone it
        reshapes whichever position it is told to read
        """
        if pad_mask is None:
            pad_mask = srcs.eq(self.pad_id)
        states = self.hidden(srcs, pad_mask)
        real = (~pad_mask).unsqueeze(-1).to(states.dtype)
        return (states * real).sum(1) / real.sum(1).clamp(min=1)

    def pooled_state(self, srcs, pad_mask=None):
        """
        whatever this model was built to pool with, so a caller reading a
        representation off the LM does not have to know which it is
        """
        if self.hparams.pooling == 'mean':
            return self.mean_state(srcs, pad_mask)
        return self.eos_state(srcs, pad_mask)

    @torch.no_grad()
    def log_prob(self, srcs, pad_mask=None):
        """
        the summed log-probability of each sequence, (batch,).

        what this means depends on how the batch was encoded. a sequence ending
        in <eos> gives log p(word): the model had to predict that the form
        stops there. a sequence encoded with eos=False gives the log prefix
        probability, i.e. the probability that a word *begins* this way, with
        every continuation summed out -- which is what scoring a bare onset
        against whole-word ratings calls for
        """
        if pad_mask is None:
            pad_mask = srcs.eq(self.pad_id)
        logits = self.forward(srcs, pad_mask)
        # column t predicts token t+1, so drop the last column and the <bos>
        scores = F.log_softmax(logits[:, :-1], dim=-1)
        tgts = srcs[:, 1:]
        token = scores.gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
        return (token * tgts.ne(self.pad_id)).sum(-1)

    @torch.no_grad()
    def generate(self, prompts, pad_mask, eos_id, max_new=32):
        """
        greedily continues each prompt until <eos>, returning a list of id
        lists holding only what was generated.

        every prompt in the batch must be the same length. positions here are
        absolute and learned, so a padded prompt would place <bos> at an index
        the model never saw it at during training; the caller batches equal
        lengths together instead of padding
        """
        self.eval()
        device = prompts.device
        done = torch.zeros(prompts.size(0), dtype=torch.bool, device=device)
        out = [[] for _ in range(prompts.size(0))]

        for _ in range(max_new):
            if prompts.size(1) >= self.max_len:
                break
            step = self.forward(prompts, pad_mask)[:, -1].argmax(-1)
            step[done] = self.pad_id
            for row in (~done).nonzero(as_tuple=True)[0].tolist():
                out[row].append(step[row].item())
            done |= step.eq(eos_id)
            if done.all():
                break
            prompts = torch.cat([prompts, step.unsqueeze(1)], dim=1)
            pad_mask = torch.cat(
                [pad_mask, torch.zeros_like(step, dtype=torch.bool).unsqueeze(1)],
                dim=1)
        return [[i for i in row if i != eos_id] for row in out]

    def step(self, batch):
        """
        the shared train/val computation: next-segment cross entropy over a
        batch dict from SegStrDataset.collate, padded targets ignored
        """
        srcs, pad_mask = batch['srcs'], batch['pad_mask']
        logits = self.forward(srcs, pad_mask)
        tgts = srcs[:, 1:]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               tgts.reshape(-1), ignore_index=self.pad_id)
        return loss

    def training_step(self, batch, batch_idx):
        """
        one optimization step; the returned loss is what Lightning backprops
        """
        loss = self.step(batch)
        # batch_size is explicit because the batch is a dict, not a tensor.
        # aggregated over the epoch rather than logged per step, so a run's
        # metrics file has one row per epoch and can be read as a curve
        self.log_dict({'train_loss': loss, 'train_ppl': loss.exp()},
                      prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=batch['srcs'].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        scores one dev batch; nothing is returned since there is no backward pass
        """
        loss = self.step(batch)
        self.log_dict({'val_loss': loss, 'val_ppl': loss.exp()},
                      prog_bar=True, batch_size=batch['srcs'].size(0))

    def loader(self, ds, shuffle):
        """
        wraps `ds` in a DataLoader carrying its collate, which is what pads a
        batch of variable-length forms; the default collate would fail on them
        """
        return DataLoader(ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          collate_fn=ds.collate,
                          drop_last=False)

    def train_dataloader(self):
        """
        the training loader, reshuffled each epoch
        """
        return self.loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        """
        the dev loader; order is irrelevant to the metrics, so don't shuffle.
        None where nothing was held out -- under k-fold the caller may hand the
        fold's own slice in, or nothing at all
        """
        return None if self.val_ds is None else self.loader(self.val_ds, False)

    def configure_optimizers(self):
        """
        the optimizer Lightning drives; return a dict to add an lr scheduler
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == '__main__':
    from SegStrDataset import SegStrDataset

    ds = SegStrDataset('../data/hayes_wilson/english.txt', form_col=1)
    model = SegStrLM(len(ds.vocab), ds.pad_id, max_len=ds.max_len)
    print(f'{len(ds):,} forms, vocab {len(ds.vocab)}, max_len {ds.max_len}')
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters')

    batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=ds.collate)))
    print(f'loss {model.step(batch).item():.4f}  '
          f'(chance {torch.tensor(len(ds.vocab)).float().log().item():.4f})')

    # the causal mask is the thing most likely to be silently wrong: a later
    # segment must not change the logits at an earlier position
    model.eval()
    with torch.no_grad():
        before = model(batch['srcs'], batch['pad_mask'])
        poked = batch['srcs'].clone()
        poked[:, -1] = (poked[:, -1] + 1) % len(ds.vocab)
        after = model(poked, batch['pad_mask'])
    print('causal (earlier logits unmoved):',
          torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6))

    # p(word) = p(prefix) * p(<eos> | prefix), so dropping <eos> can only raise
    # the score; this is the invariant onset scoring depends on
    word = torch.tensor([ds.vocab.encode(ds.forms[0])])
    prefix = torch.tensor([ds.vocab.encode(ds.forms[0], eos=False)])
    print(f'{"".join(ds.forms[0])}: log p(word) {model.log_prob(word).item():.3f} '
          f'<= log p(prefix) {model.log_prob(prefix).item():.3f}')
