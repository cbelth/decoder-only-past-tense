import random

from torch.nn import functional as F

def change(lemma, past):
    """
    the edit that takes `lemma` to `past`, as (old, new, right): strip the
    longest shared prefix, then the longest shared suffix of what remains, and
    `old` -> `new` is what is left, occurring just before the shared suffix
    `right`. this is anchored to the end of the word, which is where English
    past tense marking lives, and covers irregulars as well as regulars:

        f a ɪ n -> f a ɪ n d    ((), ('d',), ())          add d
        f l ɪ ŋ -> f l ʌ ŋ      (('ɪ',), ('ʌ',), ('ŋ',))  ɪ -> ʌ before final ŋ
        l iː v  -> l ɛ f t      (('iː', 'v'), ('ɛ', 'f', 't'), ())
    """
    lemma, past = tuple(lemma), tuple(past)
    pre = 0
    while pre < min(len(lemma), len(past)) and lemma[pre] == past[pre]:
        pre += 1
    suf = 0
    while (suf < min(len(lemma), len(past)) - pre
           and lemma[-1 - suf] == past[-1 - suf]):
        suf += 1
    return (lemma[pre:len(lemma) - suf], past[pre:len(past) - suf],
            lemma[len(lemma) - suf:])

def apply(rule, lemma):
    """
    applies a `change` rule to `lemma`, or None where the lemma does not end
    in the string the rule rewrites
    """
    old, new, right = rule
    lemma = tuple(lemma)
    context = old + right
    if len(context) > len(lemma) or lemma[len(lemma) - len(context):] != context:
        return None
    return list(lemma[:len(lemma) - len(context)] + new + right)

VOWELS = frozenset('ɪ ə æ ɛ ɹ̩ e a iː ʊ ʌ ɑː o ɔ uː'.split())

# change classes, regular first. an output's class depends only on the
# (lemma, past) pair, so a lemma's class and its neighbour's are comparable
CLASSES = (
    'reg_d',                # l a ɪ v -> l a ɪ v d
    'reg_t',                # k ʌ s -> k ʌ s t
    'reg_əd',               # w ɔ n t -> w ɔ n t ə d
    'no_change',            # h ɪ t -> h ɪ t (the sing -> sing type)
    'vowel_change',         # s ɪ ŋ -> s æ ŋ, t e ɪ k -> t ʊ k
    'vowel_change_suffix',  # k iː p -> k ɛ p t, t ɛ l -> t o ʊ l d
    'rhyme_replaced',       # θ ɪ ŋ k -> θ ɔ t, t iː t͡ʃ -> t ɔ t
    'coda_change',          # b ɪ l d -> b ɪ l t, m e ɪ k -> m e ɪ d
    'other',                # ɡ o ʊ -> w ɛ n t, and malformed outputs
)
REGULAR = frozenset(CLASSES[:3])

def final_syllable(segs) -> tuple:
    """
    splits `segs` into (before, nucleus, coda) around its last run of
    vowels, so a diphthong written e ɪ is one nucleus
    """
    segs = tuple(segs)
    end = len(segs)
    while end and segs[end - 1] not in VOWELS:
        end -= 1
    start = end
    while start and segs[start - 1] in VOWELS:
        start -= 1
    return segs[:start], segs[start:end], segs[end:]

def classify(lemma, past) -> str:
    """
    the change class of a lemma -> past pair, one of CLASSES.

    beyond the regular suffixes and no change, irregulars are told apart by
    their final syllable: whether the nucleus changed, and whether the coda
    stayed, gained a t/d, or was replaced by material ending in t/d. anything
    that alters the word before its final nucleus is `other`, which is where
    suppletion (go -> went) lands and also most malformed model outputs.

    known compromises: buy -> bought is vowel_change_suffix (no coda to
    replace), fight -> fought is vowel_change (coda already t), and
    stand -> stood and hear -> heard are rhyme_replaced
    """
    lemma, past = list(lemma), list(past)
    if past == lemma:
        return 'no_change'
    if past == lemma + ['d']:
        return 'reg_d'
    if past == lemma + ['t']:
        return 'reg_t'
    if past == lemma + ['ə', 'd']:
        return 'reg_əd'

    before, nucleus, coda = final_syllable(lemma)
    past_before, past_nucleus, past_coda = final_syllable(past)
    if before != past_before:
        return 'other'
    vowel = nucleus != past_nucleus
    if past_coda == coda:
        return 'vowel_change' if vowel else 'other'
    if past_coda in (coda + ('t',), coda + ('d',)):
        return 'vowel_change_suffix' if vowel else 'other'
    if past_coda and past_coda[-1] in ('t', 'd'):
        return 'rhyme_replaced' if vowel else 'coda_change'
    return 'other'

def is_regular(lemma, past):
    """
    whether `past` is `lemma` plus one of the regular allomorphs
    """
    return list(past) in ([*lemma, 'd'], [*lemma, 't'], [*lemma, 'ə', 'd'])

def edit_distance(a, b) -> int:
    """
    segment-level Levenshtein distance
    """
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def string_nearest(query_ds, ref_ds, seed=0) -> list:
    """
    for each query lemma, the reference lemma with the smallest edit distance,
    ties broken at random. the surface baseline: analogy over the strings
    themselves rather than over anything the model learned
    """
    rng = random.Random(seed)
    out = []
    for src in query_ds.srcs:
        dists = [edit_distance(src, ref) for ref in ref_ds.srcs]
        best = min(dists)
        out.append(rng.choice([i for i, d in enumerate(dists) if d == best]))
    return out

def analogize(query_ds, ref_ds, neighbors, preds, ref_preds) -> list:
    """
    for each query row, compares it with its neighbour `neighbors[i]` in
    `ref_ds` in two ways, taking the neighbour's past tense to be the model's
    own output for it (`ref_preds`), since that is what the model learned:

    - exact: applies the neighbour's lemma -> past change to the query lemma
      and checks the resulting form against the model's prediction and gold
    - class: checks the neighbour's change class against the class of the
      query's lemma -> prediction and lemma -> gold
    """
    rows = []
    for i, n in enumerate(neighbors):
        src, pred, gold = query_ds.srcs[i], preds[i], query_ds.tgts[i]
        nbr_src, nbr_out = ref_ds.srcs[n], ref_preds[n]
        form = apply(change(nbr_src, nbr_out), src)
        class_pred, class_gold = classify(src, pred), classify(src, gold)
        class_nbr = classify(nbr_src, nbr_out)
        rows.append({'src': src, 'gold': gold, 'pred': pred,
                     'neighbor_src': nbr_src, 'neighbor_out': nbr_out,
                     'neighbor_gold': ref_ds.tgts[n], 'analogy': form,
                     'regular': is_regular(src, gold),
                     'applies': form is not None,
                     'matches_pred': form == pred, 'matches_gold': form == gold,
                     'class_pred': class_pred, 'class_gold': class_gold,
                     'class_neighbor': class_nbr,
                     'class_match_pred': class_nbr == class_pred,
                     'class_match_gold': class_nbr == class_gold})
    return rows

def kappa(truth, guess) -> float:
    """
    Cohen's kappa between two label lists: agreement above what the two
    label distributions would give by chance. near 0 for a method that only
    ever guesses the majority class, however high its accuracy
    """
    n = len(truth)
    if not n:
        return 0.0
    observed = sum(t == g for t, g in zip(truth, guess)) / n
    expected = sum((truth.count(c) / n) * (guess.count(c) / n) for c in set(truth) | set(guess))
    return 0.0 if expected == 1 else (observed - expected) / (1 - expected)

def summarize(rows) -> dict:
    """
    exact-match and class-agreement rates, overall and by whether the gold
    form and the model's prediction are regular, irregular or `other`; per-class recall and the
    confusion matrix against the class of the model's prediction.

    exact match counts a change that does not apply as a miss, so
    `applies` is reported alongside it
    """
    def rates(subset):
        n = max(len(subset), 1)
        return {'n': len(subset),
                'applies': sum(r['applies'] for r in subset) / n,
                'matches_pred': sum(r['matches_pred'] for r in subset) / n,
                'matches_gold': sum(r['matches_gold'] for r in subset) / n,
                'class_acc_pred': sum(r['class_match_pred'] for r in subset) / n,
                'class_acc_gold': sum(r['class_match_gold'] for r in subset) / n,
                'kappa_pred': kappa([r['class_pred'] for r in subset],
                                    [r['class_neighbor'] for r in subset])}
    confusion = {c: {} for c in CLASSES}
    for r in rows:
        cell = confusion[r['class_pred']]
        cell[r['class_neighbor']] = cell.get(r['class_neighbor'], 0) + 1
    per_class = {c: {'n': sum(confusion[c].values()),
                     'recall': confusion[c].get(c, 0) / max(sum(confusion[c].values()), 1),
                     'guessed': sum(confusion[t].get(c, 0) for t in CLASSES)}
                 for c in CLASSES}
    return {'all': rates(rows),
            'gold_regular': rates([r for r in rows if r['class_gold'] in REGULAR]),
            'gold_irregular': rates([r for r in rows if r['class_gold'] not in REGULAR]),
            'pred_regular': rates([r for r in rows if r['class_pred'] in REGULAR]),
            # real irregular outputs, apart from `other`, which is mostly
            # malformed output that no neighbour could be expected to predict
            'pred_irregular': rates([r for r in rows if r['class_pred'] not in REGULAR
                                     and r['class_pred'] != 'other']),
            'pred_other': rates([r for r in rows if r['class_pred'] == 'other']),
            'per_class': per_class,
            # rows are the class of the model's prediction, columns the
            # neighbour's
            'confusion': confusion}

MODEL_POOLS = ('sep', 'mean', 'stem', 'decision')
# neighbours in the spline view of the decision step (see spline.py): the
# nearest ReLU region, and the most similar templates
SPLINE_METHODS = ('model_region', 'model_template', 'model_template_chosen')
METHODS = (tuple(f'model_{pool}' for pool in MODEL_POOLS) + SPLINE_METHODS
           + ('string', 'final', 'majority', 'random'))

def final_nearest(query_ds, ref_ds, seed=0) -> list:
    """
    for each query lemma, a random reference lemma ending in the same
    segment, or any reference lemma where none does. the regular allomorph
    is fixed by the final segment, so this is the baseline a representation
    has to beat on regulars
    """
    rng = random.Random(seed)
    by_final = {}
    for i, src in enumerate(ref_ds.srcs):
        by_final.setdefault(src[-1] if src else None, []).append(i)
    return [rng.choice(by_final.get(src[-1] if src else None, range(len(ref_ds))))
            for src in query_ds.srcs]

def majority_nearest(query_ds, ref_ds, ref_preds, seed=0) -> list:
    """
    for each query lemma, a random reference lemma from the most common
    change class among the model's outputs on `ref_ds`: the majority-class
    baseline, in the same form as the other methods
    """
    rng = random.Random(seed)
    classes = [classify(src, out) for src, out in zip(ref_ds.srcs, ref_preds)]
    top = max(set(classes), key=classes.count)
    pool = [i for i, c in enumerate(classes) if c == top]
    return [rng.choice(pool) for _ in range(len(query_ds))]

def spline_nearest(name, query, ref, seed=0) -> list:
    """
    nearest reference row under a spline method, from `spline.decision_features`
    of both sides: least Hamming distance between ReLU codes for
    `model_region`, ties broken at random since distances are integers;
    greatest template cosine for the template methods
    """
    if name == 'model_region':
        from spline import hamming
        dist = hamming(query['code'], ref['code'])
        rng = random.Random(seed)
        return [rng.choice((row == row.min()).nonzero().flatten().tolist())
                for row in dist]
    key = name[len('model_'):]
    q = F.normalize(query[key], dim=-1)
    r = F.normalize(ref[key], dim=-1)
    return (q @ r.T).argmax(1).tolist()

def neighbor_sets(model, query_ds, ref_ds, query_preds, ref_preds,
                  methods=METHODS, seed=0) -> dict:
    """
    the k=1 neighbour of every query row under each of `methods`: the model's
    representations (`model_<pool>`), its decision-step regions and templates
    (SPLINE_METHODS), plus the baselines. the preds are the model's own
    outputs, which `model_decision` and the spline methods read along
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f'unknown methods {sorted(unknown)}; choose from {METHODS}')
    sets = {}
    features = None
    for name in methods:
        if name in SPLINE_METHODS:
            if features is None:
                from spline import decision_features
                features = (decision_features(model, query_ds, query_preds),
                            decision_features(model, ref_ds, ref_preds))
            sets[name] = spline_nearest(name, *features, seed=seed)
        elif name.startswith('model_'):
            _, idxs = model.nearest(query_ds, ref_ds, k=1, pool=name[len('model_'):],
                                    query_preds=query_preds, ref_preds=ref_preds)
            sets[name] = idxs[:, 0].tolist()
        elif name == 'string':
            sets[name] = string_nearest(query_ds, ref_ds, seed=seed)
        elif name == 'final':
            sets[name] = final_nearest(query_ds, ref_ds, seed=seed)
        elif name == 'majority':
            sets[name] = majority_nearest(query_ds, ref_ds, ref_preds, seed=seed)
        else:
            rng = random.Random(seed)
            sets[name] = [rng.randrange(len(ref_ds)) for _ in range(len(query_ds))]
    return sets

def run_methods(model, query_ds, ref_ds, query_preds, ref_preds,
                methods=METHODS, seed=0) -> dict:
    """
    the analogy rows for every method, keyed by method name
    """
    sets = neighbor_sets(model, query_ds, ref_ds, query_preds, ref_preds,
                         methods=methods, seed=seed)
    return {name: analogize(query_ds, ref_ds, idxs, query_preds, ref_preds)
            for name, idxs in sets.items()}

def write_tsv(path, rows_by_method) -> None:
    """
    one line per item per method; NA where the neighbour's change does not
    apply to the item
    """
    cols = ['neighbors', 'src', 'gold', 'pred', 'neighbor_src', 'neighbor_out',
            'neighbor_gold', 'analogy', 'matches_pred', 'matches_gold',
            'class_gold', 'class_pred', 'class_neighbor']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for name, rows in rows_by_method.items():
            for r in rows:
                f.write('\t'.join([name] + [
                    ' '.join(r[c]) for c in ('src', 'gold', 'pred', 'neighbor_src',
                                             'neighbor_out', 'neighbor_gold')] + [
                    ' '.join(r['analogy']) if r['applies'] else 'NA',
                    str(int(r['matches_pred'])), str(int(r['matches_gold'])),
                    r['class_gold'], r['class_pred'], r['class_neighbor']]) + '\n')
