import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from SegVocab import SegVocab

class SegStrPairDataset(Dataset):
    """
    a tsv of transduction triples -- morphosyntactic features, a source form and
    the target form they call for -- encoded for a decoder-only model as one
    sequence:

        <bos> <N> <ACC> <PL>  k i t a p  <sep>  k i t a p l a ɾ ɯ  <eos>
        |------------- prompt --------------|  |-------- target --------|

    a causal LM over this computes p(target | features, source) by the chain
    rule, so no encoder-decoder is needed and SegStrLM runs on it unchanged.
    the whole source is left context for every target position, which is what
    makes a right-conditioned process expressible even under a causal mask.

    features are decomposed into one token each rather than one token per
    bundle. an atomic N;ACC;PL shares nothing with N;ACC;SG, so the model would
    have to learn every cell independently; separate tags let it learn what ACC
    does and what PL does, which is also what makes holding out a whole cell a
    meaningful test.

    tags are bracketed to keep them out of the segment inventory: the noun tag
    N would otherwise be one lowercase away from the segment n.
    """

    def __init__(self, path, vocab=None, feats_col=2, src_col=0, tgt_col=1,
                 rows=None, header=False):
        """
        reads `path`, or takes `rows` already read, and encodes each triple.
        pass the training dataset's `vocab` to anything evaluated against it
        """
        self.path = path
        self.feats_col = feats_col
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.header = header
        self.load(vocab, rows)

    def load(self, vocab=None, rows=None) -> None:
        """
        reads the triples and settles the vocab over both segments and tags
        """
        if rows is None:
            with open(self.path, 'r', encoding='utf-8') as f:
                if self.header:
                    next(f)
                rows = [line.rstrip('\n').split('\t') for line in f if line.strip()]
        self.rows = rows

        self.feats = [tuple(f'<{tag}>' for tag in row[self.feats_col].split(';'))
                      for row in rows]
        self.srcs = [row[self.src_col].split() for row in rows]
        self.tgts = [row[self.tgt_col].split() for row in rows]

        self.vocab = vocab if vocab is not None else SegVocab.from_sequences(
            [list(f) + s + t for f, s, t in zip(self.feats, self.srcs, self.tgts)])

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id

    @property
    def max_len(self) -> int:
        """
        the longest encoded triple, so a model can size its positional table
        from the data rather than a guess
        """
        return max(len(f) + len(s) + len(t) for f, s, t
                   in zip(self.feats, self.srcs, self.tgts)) + 3

    def unknown(self) -> set:
        """
        tokens this file uses that the vocab has no id for
        """
        return self.vocab.unknown([list(f) + s + t for f, s, t
                                   in zip(self.feats, self.srcs, self.tgts)])

    def prompt(self, idx) -> list:
        """
        the ids of everything given at test time, up to and including <sep>
        """
        return ([self.vocab.bos_id]
                + self.vocab.encode(list(self.feats[idx]) + self.srcs[idx],
                                    bos=False, eos=False)
                + [self.vocab.sep_id])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx) -> dict:
        """
        the full sequence and how much of it is prompt, which is what lets the
        loss be taken over the target alone
        """
        prompt = self.prompt(idx)
        target = self.vocab.encode(self.tgts[idx], bos=False, eos=True)
        return {'src': torch.tensor(prompt + target, dtype=torch.long),
                'prompt_len': torch.tensor(len(prompt), dtype=torch.long)}

    def collate(self, batch) -> dict:
        """
        pads a batch and carries the prompt lengths alongside it
        """
        srcs = pad_sequence([item['src'] for item in batch],
                            batch_first=True, padding_value=self.pad_id)
        # True marks padding, matching nn.TransformerEncoder's src_key_padding_mask
        return {'srcs': srcs, 'pad_mask': srcs.eq(self.pad_id),
                'prompt_lens': torch.stack([item['prompt_len'] for item in batch])}

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = SegStrPairDataset('../data/en_0_100.train')
    print(f'{len(ds):,} triples, vocab {len(ds.vocab)}, max_len {ds.max_len}')
    print('tags in the vocab:', [t for t in ds.vocab.itos if t.startswith('<')])

    item = ds[0]
    print('\n', ds.rows[0][0], ds.rows[0][2], '->', ds.rows[0][1])
    print('  encoded:', ' '.join(ds.vocab.itos[i] for i in item['src'].tolist()))
    print('  prompt_len:', item['prompt_len'].item(),
          '| prompt ends at:', ds.vocab.itos[item['src'][item['prompt_len'] - 1]])

    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=ds.collate)))
    print('\n batch:', {k: tuple(v.shape) for k, v in batch.items()})

    # a vocab built from train alone misses segments the eval splits use
    gold = SegStrPairDataset('../data/en_0.gold', vocab=ds.vocab)
    print(' gold segments unknown to the train vocab:', gold.unknown() or 'none')
