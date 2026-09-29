import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import lightning as L

def copied(lemma, pred) -> list:
    """
    the longest prefix `pred` shares with `lemma`: what the model copied
    before its first departure from the lemma
    """
    n = 0
    while n < min(len(lemma), len(pred)) and lemma[n] == pred[n]:
        n += 1
    return list(lemma[:n])

class SegStrTransducer(L.LightningModule):
    """
    a SegStrLM trained to map (features, source) to a target form, with the
    loss taken over the target alone.

    the prompt -- the feature tags and the source, up to <sep> -- is given at
    test time, so predicting it is not the task. the tags in particular carry
    no signal: in a rectangular paradigm table every cell occurs once per
    lemma, so their distribution is uniform and a model scored on them learns a
    constant. masking is per-position and nothing about the model changes; it
    is the same causal stack under the same next-segment objective, told which
    predictions to count.

    keeping the loss on the source instead would add a language model over the
    source forms, which is real phonotactic signal and worth having where the
    labelled set is small. that is `source_weight`, off by default here.
    """

    def __init__(self, lm, lr=1e-3, source_weight=0.0,
                 train_ds=None, val_ds=None, batch_size=64, num_workers=0):
        """
        wraps `lm`, which may be freshly built or already trained
        """
        super().__init__()
        # the LM is a module and the datasets are data; neither is an hparam
        self.save_hyperparameters(ignore=['lm', 'train_ds', 'val_ds'])
        self.lm = lm
        self.train_ds = train_ds
        self.val_ds = val_ds

    def forward(self, srcs, pad_mask=None):
        """
        next-token logits over the whole sequence, (batch, seq, vocab)
        """
        return self.lm(srcs, pad_mask)

    def targets(self, srcs, prompt_lens, keep_prompt=False):
        """
        the labels for a batch: position t predicts token t+1, so the target
        row is `srcs` shifted left, with everything the prompt covers set to
        the pad id and thereby ignored
        """
        tgts = srcs[:, 1:].clone()
        if not keep_prompt:
            # token t+1 belongs to the prompt while t+1 < prompt_len
            index = torch.arange(tgts.size(1), device=srcs.device)
            tgts[index.unsqueeze(0) < (prompt_lens.unsqueeze(1) - 1)] = self.lm.pad_id
        return tgts

    def step(self, batch):
        """
        the shared train/val computation: cross entropy over target positions,
        and the fraction of target tokens predicted correctly
        """
        srcs, pad_mask, prompt_lens = (batch['srcs'], batch['pad_mask'],
                                       batch['prompt_lens'])
        logits = self.forward(srcs, pad_mask)[:, :-1]
        tgts = self.targets(srcs, prompt_lens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgts.reshape(-1), ignore_index=self.lm.pad_id)

        scored = tgts.ne(self.lm.pad_id)
        correct = logits.argmax(-1).eq(tgts) & scored
        acc = correct.sum() / scored.sum().clamp(min=1)

        if self.hparams.source_weight:
            whole = self.targets(srcs, prompt_lens, keep_prompt=True)
            loss = loss + self.hparams.source_weight * F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), whole.reshape(-1),
                ignore_index=self.lm.pad_id)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """
        one optimization step; the returned loss is what Lightning backprops
        """
        loss, acc = self.step(batch)
        self.log_dict({'train_loss': loss, 'train_tok_acc': acc},
                      prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=batch['srcs'].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        scores one dev batch; nothing is returned since there is no backward pass
        """
        loss, acc = self.step(batch)
        self.log_dict({'val_loss': loss, 'val_tok_acc': acc},
                      prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=batch['srcs'].size(0))

    @torch.no_grad()
    def predict(self, ds, max_new=None) -> list:
        """
        greedily decodes a target for every row of `ds`, in order, as segment
        lists. prompts are grouped by length rather than padded, since
        SegStrLM.generate needs every prompt in a batch to be the same length
        """
        self.eval()
        max_new = max_new or self.lm.max_len
        by_len = {}
        for idx in range(len(ds)):
            by_len.setdefault(len(ds.prompt(idx)), []).append(idx)

        preds = [None] * len(ds)
        size = self.hparams.batch_size
        for group in by_len.values():
            for start in range(0, len(group), size):
                rows = group[start:start + size]
                prompts = torch.tensor([ds.prompt(idx) for idx in rows],
                                       dtype=torch.long, device=self.device)
                outs = self.lm.generate(prompts, prompts.eq(self.lm.pad_id),
                                        ds.vocab.eos_id, max_new=max_new)
                for idx, out in zip(rows, outs):
                    preds[idx] = ds.vocab.decode(out)
        return preds

    def evaluate(self, ds) -> dict:
        """
        exact-match word accuracy of greedy decoding against the targets in
        `ds`, alongside the predictions themselves
        """
        preds = self.predict(ds)
        correct = [pred == tgt for pred, tgt in zip(preds, ds.tgts)]
        return {'acc': sum(correct) / max(len(correct), 1), 'preds': preds,
                'correct': correct}

    @torch.no_grad()
    def represent(self, ds, pool='sep', preds=None) -> torch.Tensor:
        """
        a (len(ds), d_model) representation of each row's prompt -- the lemma
        and its tags, never the target -- read off the final layer.

        `pool='sep'` takes the state at <sep>, the last prompt position and so,
        under the causal mask, the only one that has seen the whole lemma; it
        is also the state the first target segment is predicted from.
        `pool='mean'` averages every prompt position instead, i.e. the lemma's
        prefixes (see SegStrLM.mean_state).

        `pool='stem'` appends an unchanged copy of the lemma after <sep> and
        takes the state at its last segment: the point where the model
        chooses what, if anything, follows the stem. it is built from the
        lemma alone, so it treats train and test rows alike; for irregulars
        it is a state the model never reached in training, since it would
        have altered the stem before getting there.

        `pool='decision'` follows the model's own output, `preds` (decoded
        here if not given), for as long as it copies the lemma, and takes the
        state at the first step where it departs: the state that chose the
        suffix, the changed vowel or <eos>. f a ɪ n -> f a ɪ n d reads the
        state after the whole stem, as `stem` does; s ɪ ŋ -> s æ ŋ reads the
        state after s, which chose æ over ɪ; ɡ oʊ -> w ɛ n t reads <sep>. every
        input up to there is a copied prefix of the lemma, so the output
        decides where the state is read but is never inside it.

        sequences are right-padded, which the causal mask keeps from reaching
        any real position
        """
        if pool not in ('sep', 'mean', 'stem', 'decision'):
            raise ValueError("pool must be 'sep', 'mean', 'stem' or 'decision', "
                             f"not {pool!r}")
        if pool == 'decision' and preds is None:
            preds = self.predict(ds)
        self.eval()
        prompts = []
        for idx in range(len(ds)):
            ids = ds.prompt(idx)
            if pool == 'stem':
                ids = ids + ds.vocab.encode(ds.srcs[idx], bos=False, eos=False)
            elif pool == 'decision':
                ids = ids + ds.vocab.encode(copied(ds.srcs[idx], preds[idx]),
                                            bos=False, eos=False)
            prompts.append(torch.tensor(ids, dtype=torch.long))
        reps = []
        size = self.hparams.batch_size
        for start in range(0, len(prompts), size):
            batch = pad_sequence(prompts[start:start + size], batch_first=True,
                                 padding_value=self.lm.pad_id).to(self.device)
            pad_mask = batch.eq(self.lm.pad_id)
            states = self.lm.hidden(batch, pad_mask)
            if pool == 'mean':
                real = (~pad_mask).unsqueeze(-1).to(states.dtype)
                reps.append((states * real).sum(1) / real.sum(1))
            else:
                last = (~pad_mask).sum(1) - 1
                reps.append(states[torch.arange(states.size(0), device=states.device), last])
        return torch.cat(reps)

    @torch.no_grad()
    def nearest(self, query_ds, ref_ds, k=1, pool='sep', query_preds=None,
                ref_preds=None) -> tuple:
        """
        for each row of `query_ds`, the `k` rows of `ref_ds` whose
        representations are most cosine-similar, as (sims, indices), both
        (len(query_ds), k) and best first. the preds are the model's own
        outputs, used by `pool='decision'` and decoded if not given
        """
        query = F.normalize(self.represent(query_ds, pool, query_preds), dim=-1)
        ref = F.normalize(self.represent(ref_ds, pool, ref_preds), dim=-1)
        return (query @ ref.T).topk(k, dim=-1)

    def loader(self, ds, shuffle):
        """
        wraps `ds` in a DataLoader carrying its collate, which pads the batch
        """
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          collate_fn=ds.collate, drop_last=False)

    def train_dataloader(self):
        """
        the training loader, reshuffled each epoch
        """
        return self.loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        """
        the dev loader; order is irrelevant to the metrics, so don't shuffle.
        None where nothing was held out
        """
        return None if self.val_ds is None else self.loader(self.val_ds, False)

    def configure_optimizers(self):
        """
        the optimizer Lightning drives
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == '__main__':
    from SegStrPairDataset import SegStrPairDataset
    from SegStrLM import SegStrLM

    ds = SegStrPairDataset('../data/en_0_100.train')
    lm = SegStrLM(len(ds.vocab), ds.pad_id, max_len=ds.max_len)
    model = SegStrTransducer(lm)

    batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=ds.collate)))
    loss, acc = model.step(batch)
    print(f'{len(ds):,} triples | loss {loss.item():.4f} | token acc {acc.item():.3f}')

    # the mask must cover exactly the prompt: every scored position should sit
    # at or past <sep>, and none before it
    tgts = model.targets(batch['srcs'], batch['prompt_lens'])
    scored = tgts.ne(ds.pad_id)
    first = scored.float().argmax(1)
    print('first scored position == prompt_len - 1 for every row:',
          bool(first.eq(batch['prompt_lens'] - 1).all()))
    print('scored tokens per row:', scored.sum(1).tolist())
    print('target lengths + eos:', [len(t) + 1 for t in ds.tgts[:8]])

    # untrained, so accuracy should be ~0; this checks decoding runs end to end
    result = model.evaluate(ds)
    print(f'untrained word acc {result["acc"]:.3f} | first pred:',
          ' '.join(result['preds'][0]))
