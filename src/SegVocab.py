import json

PAD, UNK, BOS, EOS, SEP = '<pad>', '<unk>', '<bos>', '<eos>', '<sep>'
SPECIALS = (PAD, UNK, BOS, EOS, SEP)

class SegVocab:
    """
    maps phonological segments to ids. built once from the training data and
    then passed to every dataset that has to agree with it: a vocab built
    per-file would give the same segment different ids in the lexicon and in
    the wug list, which trains and evaluates in two unrelated id spaces.

    <bos> and <eos> bracket a form. <eos> is what makes the LM model word
    length -- without it the probabilities do not normalise over strings, so
    p(word) is not comparable across lengths -- and it doubles as the position
    a classifier head pools, since under a causal mask the last position is the
    only one that has seen the whole form.
    """

    def __init__(self, segs):
        """
        builds the vocab over `segs`, which is sorted so that ids depend on the
        segment inventory alone and not on corpus order
        """
        self.itos = list(SPECIALS) + sorted(set(segs) - set(SPECIALS))
        self.stoi = {seg: idx for idx, seg in enumerate(self.itos)}

    @classmethod
    def from_sequences(cls, seqs):
        """
        builds a vocab from an iterable of segment sequences
        """
        return cls({seg for seq in seqs for seg in seq})

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def sep_id(self) -> int:
        """
        divides a transduction prompt from the form it should produce
        """
        return self.stoi[SEP]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, seg) -> bool:
        return seg in self.stoi

    def encode(self, segs, bos=True, eos=True) -> list:
        """
        maps a segment sequence to ids, unseen segments falling back to <unk>.
        `eos` is switched off to score a form as a prefix rather than a whole
        word, which is what marginalising over continuations amounts to
        """
        ids = [self.stoi.get(seg, self.unk_id) for seg in segs]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_specials=True) -> list:
        """
        maps ids back to segments, by default dropping the brackets and padding
        """
        segs = [self.itos[idx] for idx in ids]
        if strip_specials:
            segs = [seg for seg in segs if seg not in SPECIALS]
        return segs

    def unknown(self, seqs) -> set:
        """
        the segments in `seqs` this vocab has no id for. worth checking before
        evaluating on a file the vocab was not built from, since the fallback
        to <unk> is silent
        """
        return {seg for seq in seqs for seg in seq if seg not in self.stoi}

    def save(self, path) -> None:
        """
        writes the segment inventory, so a later run can score against the same
        ids the model was trained with
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """
        reads a vocab written by `save`, preserving the stored id order
        """
        with open(path, 'r', encoding='utf-8') as f:
            itos = json.load(f)
        vocab = cls.__new__(cls)
        vocab.itos = itos
        vocab.stoi = {seg: idx for idx, seg in enumerate(itos)}
        return vocab

if __name__ == '__main__':
    lex = '../data/hayes_hungarian/hungarian_hnc.txt'
    wugs = '../data/hayes_hungarian/hayes_wuglist.txt'

    with open(lex, encoding='utf-8') as f:
        forms = [line.split('\t')[1].split() for line in f if line.strip()]
    vocab = SegVocab.from_sequences(forms)
    print(len(vocab), 'ids:', vocab.itos[:4], '...', vocab.itos[4:9])

    print(forms[0], '->', vocab.encode(forms[0]))
    print('no eos ->', vocab.encode(forms[0], eos=False))
    print('round trip:', vocab.decode(vocab.encode(forms[0])) == forms[0])

    # the case the shared vocab exists for: the wug list has to encode against
    # the lexicon's ids, so anything it uses that the lexicon lacks matters
    with open(wugs, encoding='utf-8') as f:
        nonces = [line.split() for line in f if line.strip()]
    print(f'{len(nonces)} wugs, segments unknown to the lexicon vocab:',
          vocab.unknown(nonces) or 'none')
