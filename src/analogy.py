import random

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

def analogize(query_ds, ref_ds, neighbors, preds) -> list:
    """
    for each query row, applies its neighbour's lemma -> past change to the
    query lemma, and records whether that analogical form matches the model's
    prediction and the gold form. `neighbors[i]` indexes `ref_ds`
    """
    rows = []
    for i, n in enumerate(neighbors):
        form = apply(change(ref_ds.srcs[n], ref_ds.tgts[n]), query_ds.srcs[i])
        rows.append({'src': query_ds.srcs[i], 'gold': query_ds.tgts[i],
                     'pred': preds[i], 'neighbor_src': ref_ds.srcs[n],
                     'neighbor_tgt': ref_ds.tgts[n], 'analogy': form,
                     'regular': is_regular(query_ds.srcs[i], query_ds.tgts[i]),
                     'applies': form is not None,
                     'matches_pred': form == preds[i],
                     'matches_gold': form == query_ds.tgts[i]})
    return rows

def summarize(rows) -> dict:
    """
    match rates overall and by whether the gold form is regular. a rule that
    does not apply counts as a miss, so applicability is reported alongside
    """
    def rates(subset):
        n = max(len(subset), 1)
        return {'n': len(subset),
                'applies': sum(r['applies'] for r in subset) / n,
                'matches_pred': sum(r['matches_pred'] for r in subset) / n,
                'matches_gold': sum(r['matches_gold'] for r in subset) / n}
    return {'all': rates(rows),
            'regular': rates([r for r in rows if r['regular']]),
            'irregular': rates([r for r in rows if not r['regular']])}

def neighbor_sets(model, query_ds, ref_ds, pools=('sep', 'mean', 'stem'), seed=0):
    """
    the k=1 neighbour of every query row under each of the model's
    representations, plus the surface and random baselines
    """
    sets = {}
    for pool in pools:
        _, idxs = model.nearest(query_ds, ref_ds, k=1, pool=pool)
        sets[f'model_{pool}'] = idxs[:, 0].tolist()
    sets['string'] = string_nearest(query_ds, ref_ds, seed=seed)
    rng = random.Random(seed)
    sets['random'] = [rng.randrange(len(ref_ds)) for _ in range(len(query_ds))]
    return sets
