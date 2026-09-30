"""
a transduction walked through the model's representation space: each output
segment of a held-out verb is produced by finding the nearest step of any
training verb and applying that step's edit operation to the held-out lemma.

every training verb's own output (the model's, not gold) is aligned with its
lemma by edit distance, which labels each generation step with an operation
relative to a pointer into the lemma:

    COPY       emit the lemma segment at the pointer, advance
    SUB y      emit y in place of the segment at the pointer, advance
    INS y      emit y, do not advance
    STOP       end the word

with any lemma segments skipped before the step attached to it as `skip`.
operations are relative to the lemma, so an operation taken from `s ɪ ŋ ->
s æ ŋ` (SUB æ after one COPY) can apply to `s t ɪ ŋ`, which a token could not.

steps are keyed by a representation of the prefix generated so far:

    jacobian   the Jacobian of every logit at the step through the frozen
               (piecewise affine) network, windowed as in spline.py and
               count-sketched; it does not depend on which token the model
               chooses there
    hidden     the final-layer state at the step, from which the logits are
               a linear function, so it partly encodes the choice
    context    no model: the last two output segments and the next two lemma
               segments, matched field by field; the symbolic baseline
"""
import argparse
import random

import torch
from torch.func import jacrev
from torch.nn import functional as F

import analogy
import spline

REPS = ('jacobian', 'hidden', 'context')

def align(lemma, out) -> list:
    """
    the edit operations that take `lemma` to `out`, one per generation step
    (len(out) + 1 of them, the last STOP), as (skip, kind, token) tuples.
    Levenshtein alignment, backtraced from the end of the word with ties
    resolved match, then deletion, then substitution, then insertion. that
    keeps deletions at the right edge, where English past tense changes
    happen: θ ɪ ŋ k -> θ ɔ t is ɪ -> ɔ, ŋ -> t, k dropped, not a dropped ɪ
    followed by ŋ -> ɔ and k -> t
    """
    n, m = len(lemma), len(out)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (lemma[i - 1] != out[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    pairs, i, j = [], n, m
    while i or j:
        if i and j and lemma[i - 1] == out[j - 1] and d[i][j] == d[i - 1][j - 1]:
            pairs.append(('M', out[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append(('D', None))
            i -= 1
        elif i and j and d[i][j] == d[i - 1][j - 1] + 1:
            pairs.append(('S', out[j - 1]))
            i, j = i - 1, j - 1
        else:
            pairs.append(('I', out[j - 1]))
            j -= 1
    ops, skip = [], 0
    for kind, tok in reversed(pairs):
        if kind == 'D':
            skip += 1
            continue
        ops.append((skip, {'M': 'COPY', 'S': 'SUB', 'I': 'INS'}[kind],
                    None if kind == 'M' else tok))
        skip = 0
    return ops + [(0, 'STOP', None)]

def step(op, lemma, j):
    """
    applies `op` at pointer `j`: (emitted segment or None, new pointer, done,
    valid). a COPY past the end of the lemma is invalid and ends the walk
    """
    skip, kind, tok = op
    j = min(j + skip, len(lemma))
    if kind == 'STOP':
        return None, j, True, True
    if kind == 'COPY':
        if j >= len(lemma):
            return None, j, True, False
        return lemma[j], j + 1, False, True
    if kind == 'SUB':
        return tok, min(j + 1, len(lemma)), False, True
    return tok, j, False, True

def pointers(lemma, ops) -> list:
    """
    the lemma pointer before each step when `ops` are replayed
    """
    out, j = [], 0
    for op in ops:
        out.append(j)
        _, j, done, _ = step(op, lemma, j)
        if done:
            break
    return out

class Sketch:
    """
    a count sketch to `dim` dimensions: each input coordinate is added, with a
    random sign, into one random bucket. preserves inner products in
    expectation, and needs no dense projection matrix for the ~35k-dimensional
    Jacobian windows
    """
    def __init__(self, n_in, dim=4096, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.bucket = torch.randint(0, dim, (n_in,), generator=g)
        self.sign = torch.randint(0, 2, (n_in,), generator=g).float() * 2 - 1
        self.dim = dim

    def __call__(self, x):
        return torch.zeros(self.dim).index_add_(0, self.bucket, x * self.sign)

class Representer:
    """
    the representation of a generated prefix at its last position
    """
    def __init__(self, model, vocab, rep, k=6, dim=4096):
        self.lm = model.lm.eval()
        self.vocab = vocab
        self.rep = rep
        self.k = k
        n_in = 2 * k * len(vocab) * self.lm.hparams.d_model
        self.sketch = Sketch(n_in, dim) if rep == 'jacobian' else None

    def __call__(self, ids):
        ids = torch.tensor(ids)
        t = len(ids) - 1
        if self.rep == 'hidden':
            with torch.no_grad():
                return self.lm.hidden(ids.unsqueeze(0))[0, t]
        x0 = spline.embed(self.lm, ids).detach()
        store = {}
        with torch.no_grad():
            spline.forward(self.lm, x0, store=store)
        jac = jacrev(lambda x: spline.forward(self.lm, x, frozen=store)[t])(x0)
        jac = jac.permute(1, 0, 2).detach()
        sep = ids.tolist().index(self.vocab.sep_id)
        win = torch.cat([spline.window(jac, t, self.k), spline.window(jac, sep - 1, self.k)])
        return self.sketch(win)

def context_key(lemma, out_so_far, j):
    return (tuple(out_so_far[-2:]), tuple(lemma[j:j + 2]), j >= len(lemma))

class Memory:
    """
    every generation step of every training verb along the model's own
    output: its representation (or symbolic context) and its operation
    """
    def __init__(self, model, ds, preds, rep, representer=None, seed=0):
        self.rep = rep
        self.ops, self.keys, vecs = [], [], []
        for idx in range(len(ds)):
            lemma, out = ds.srcs[idx], preds[idx]
            ops = align(lemma, out)
            ptr = pointers(lemma, ops)
            for s, op in enumerate(ops):
                self.ops.append(op)
                if rep == 'context':
                    self.keys.append(context_key(lemma, out[:s], ptr[s]))
                else:
                    ids = ds.prompt(idx) + ds.vocab.encode(out[:s], bos=False, eos=False)
                    vecs.append(representer(ids))
        if vecs:
            self.bank = F.normalize(torch.stack(vecs), dim=-1)
        self.rng = random.Random(seed)

    def nearest(self, vec=None, key=None):
        """
        the operation of the nearest stored step
        """
        if self.rep == 'context':
            scores = [sum(a == b for a, b in zip(key, k)) for k in self.keys]
            best = max(scores)
            return self.ops[self.rng.choice([i for i, s in enumerate(scores) if s == best])]
        return self.ops[(self.bank @ F.normalize(vec, dim=-1)).argmax().item()]

def walk(memory, representer, ds, idx, max_extra=5):
    """
    generates row `idx` of `ds` step by step from nearest-neighbour
    operations. returns (output, valid)
    """
    lemma, out, j = ds.srcs[idx], [], 0
    for _ in range(len(lemma) + max_extra):
        if memory.rep == 'context':
            op = memory.nearest(key=context_key(lemma, out, j))
        else:
            ids = ds.prompt(idx) + ds.vocab.encode(out, bos=False, eos=False)
            op = memory.nearest(vec=representer(ids))
        seg, j, done, valid = step(op, lemma, j)
        if not valid:
            return out, False
        if done:
            return out, True
        out.append(seg)
    return out, False

def forced(memory, representer, ds, idx, pred):
    """
    step-by-step agreement along the model's own output: at each step, does
    the nearest stored step's operation equal the model's? no errors
    compound, so this isolates each decision
    """
    lemma = ds.srcs[idx]
    ops = align(lemma, pred)
    ptr = pointers(lemma, ops)
    rows = []
    for s, op in enumerate(ops):
        if memory.rep == 'context':
            guess = memory.nearest(key=context_key(lemma, pred[:s], ptr[s]))
        else:
            ids = ds.prompt(idx) + ds.vocab.encode(pred[:s], bos=False, eos=False)
            guess = memory.nearest(vec=representer(ids))
        rows.append((op, guess))
    return rows

def evaluate(model, splits, preds, reps=REPS, eval_splits=('ftune', 'dev', 'test')):
    """
    builds each memory from the training split and scores the walk and the
    forced step agreement on the held-out splits. returns per-verb rows
    """
    train = splits['train']
    # the alignment must reproduce every training output exactly, or the
    # operations mean nothing
    for idx in range(len(train)):
        lemma, out, j, gen = train.srcs[idx], preds['train'][idx], 0, []
        for op in align(lemma, out):
            seg, j, done, _ = step(op, lemma, j)
            if done:
                break
            gen.append(seg)
        assert gen == list(out), (lemma, out, gen)

    results = []
    for rep in reps:
        representer = None if rep == 'context' else Representer(model, train.vocab, rep)
        memory = Memory(model, train, preds['train'], rep, representer)
        for split in eval_splits:
            ds = splits[split]
            for idx in range(len(ds)):
                pred = preds[split][idx]
                out, valid = walk(memory, representer, ds, idx)
                results.append({
                    'rep': rep, 'split': split, 'src': ds.srcs[idx], 'gold': ds.tgts[idx],
                    'pred': pred, 'walk': out, 'valid': valid,
                    'cls': analogy.classify(ds.srcs[idx], pred),
                    'steps': forced(memory, representer, ds, idx, pred)})
    return results

def report(results):
    reps = list(dict.fromkeys(r['rep'] for r in results))
    groups = [('regular', lambda c: c in analogy.REGULAR)] + [
        (c, lambda c, k=c: c == k) for c in analogy.CLASSES[3:]]
    print('free walk: reproduces the model\'s output exactly, by class of that output')
    print(f'  {"":22s}{"n":>6s}' + ''.join(f'{r:>10s}' for r in reps))
    for name, test in [('all', lambda c: True)] + groups:
        cells, n = [], 0
        for rep in reps:
            sub = [r for r in results if r['rep'] == rep and test(r['cls'])]
            n = len(sub)
            cells.append(sum(r['walk'] == list(r['pred']) for r in sub) / max(n, 1))
        print(f'  {name:22s}{n:6d}' + ''.join(f'{c:10.3f}' for c in cells))
    print('  ' + ' ' * 22 + ' ' * 6 + ''.join(
        f'{sum(r["walk"] == list(r["gold"]) for r in results if r["rep"] == rep) / sum(r["rep"] == rep for r in results):10.3f}'
        for rep in reps) + '   <- matches gold instead')

    print('\nforced steps: nearest step\'s operation equals the model\'s, by the model\'s operation')
    kinds = ['COPY', 'SUB', 'INS', 'STOP', 'decision steps']
    print(f'  {"":22s}{"n":>6s}' + ''.join(f'{r:>10s}' for r in reps))
    for kind in kinds:
        cells, n = [], 0
        for rep in reps:
            pairs = [(op, g) for r in results if r['rep'] == rep for op, g in r['steps']]
            if kind == 'decision steps':
                # every step that is not a plain copy with nothing skipped
                pairs = [(op, g) for op, g in pairs if op != (0, 'COPY', None)]
            else:
                pairs = [(op, g) for op, g in pairs if op[1] == kind]
            n = len(pairs)
            cells.append(sum(op == g for op, g in pairs) / max(n, 1))
        print(f'  {kind:22s}{n:6d}' + ''.join(f'{c:10.3f}' for c in cells))

def main():
    parser = argparse.ArgumentParser(
        description='walk held-out verbs through nearest-neighbour operations '
                    'in the model\'s representation space')
    parser.add_argument('--run', default='../results/en_0_1500')
    parser.add_argument('--reps', nargs='*', default=list(REPS), choices=REPS)
    parser.add_argument('--out', default=None,
                        help='TSV of per-verb walks (default: <run>/walk.tsv)')
    args = parser.parse_args()

    import run
    model, splits, _ = run.load_run(args.run)
    preds = {split: model.predict(ds) for split, ds in splits.items()}
    results = evaluate(model, splits, preds, reps=args.reps)
    report(results)
    path = args.out or f'{args.run}/walk.tsv'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('rep\tsplit\tsrc\tgold\tpred\twalk\tvalid\tclass\tsteps_agree\tsteps\n')
        for r in results:
            agree = sum(op == g for op, g in r['steps'])
            f.write('\t'.join([r['rep'], r['split'], ' '.join(r['src']), ' '.join(r['gold']),
                               ' '.join(r['pred']), ' '.join(r['walk']), str(int(r['valid'])),
                               r['cls'], str(agree), str(len(r['steps']))]) + '\n')
    print(f'\nwrote {path}')

if __name__ == '__main__':
    main()
