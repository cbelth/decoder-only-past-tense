"""
the spline view of a trained transducer (Balestriero & Baraniuk 2018, "A
Spline Theory of Deep Networks"): with its attention weights and LayerNorm
scales held at their values for a given input, the network is piecewise
affine in its input embeddings, the pieces indexed by which ReLUs are on.
within a piece every logit is an inner product with a template plus a bias,

    logit_c = sum_s <A_c[s], X0[s]> + b_c

so "which template produced this output" has an exact answer. this module
computes, at the decision step (see SegStrTransducer.represent), each verb's
ReLU pattern (its region) and templates, and reports how finely the regions
partition the verbs.
"""
import argparse
import math
import random

import torch
from torch.func import jacrev
from torch.nn import functional as F

import analogy
from SegStrTransducer import copied

def layer_norm(x, norm, frozen=None, key=None, store=None):
    """
    LayerNorm with its scale 1/sigma taken from `frozen[key]` if given, else
    computed and, when `store` is given, detached and saved there. the mean
    subtraction is linear, so with sigma fixed this is affine in x
    """
    mu = x.mean(-1, keepdim=True)
    if frozen is not None:
        sigma = frozen[key]
    else:
        sigma = torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True) + norm.eps)
        if store is not None:
            sigma = sigma.detach()
            store[key] = sigma
    return (x - mu) / sigma * norm.weight + norm.bias

def forward(lm, x0, frozen=None, store=None):
    """
    SegStrLM's forward pass written out by hand for one unpadded sequence,
    x0 the (T, d_model) summed segment and position embeddings.

    three modes. plain (neither argument): the model as it is, which must
    reproduce lm.forward. storing (`store` a dict): the same values, but the
    attention weights, LayerNorm scales and ReLU masks are detached and saved
    in `store`, so gradients see an affine function. frozen (`frozen` a dict
    from a storing pass): those saved quantities are used instead of being
    recomputed, which is the affine map itself and can be evaluated at any
    x0, including 0 for the bias
    """
    T = x0.size(0)
    causal = torch.ones(T, T, dtype=torch.bool, device=x0.device).triu(1)
    x = x0
    for i, layer in enumerate(lm.encoder.layers):
        attn = layer.self_attn
        heads, d = attn.num_heads, attn.embed_dim
        z = layer_norm(x, layer.norm1, frozen, f'ln1_{i}', store)
        q, k, v = F.linear(z, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.view(T, heads, d // heads).transpose(0, 1) for t in (q, k, v))
        if frozen is not None:
            p = frozen[f'attn_{i}']
        else:
            scores = q @ k.transpose(-1, -2) / math.sqrt(d // heads)
            p = scores.masked_fill(causal, float('-inf')).softmax(-1)
            if store is not None:
                p = p.detach()
                store[f'attn_{i}'] = p
        out = (p @ v).transpose(0, 1).reshape(T, d)
        x = x + attn.out_proj(out)

        z = layer_norm(x, layer.norm2, frozen, f'ln2_{i}', store)
        pre = layer.linear1(z)
        if frozen is not None:
            mask = frozen[f'relu_{i}']
        else:
            mask = (pre > 0).to(pre.dtype)
            if store is not None:
                store[f'relu_{i}'] = mask.detach()
        x = x + layer.linear2(pre * mask)

    h = layer_norm(x, lm.encoder.norm, frozen, 'ln_final', store)
    return lm.lm_head(h)

def embed(lm, ids):
    """
    the input the templates are over: segment plus position embeddings, with
    no dropout
    """
    pos = torch.arange(len(ids), device=ids.device)
    return lm.seg_emb(ids) + lm.pos_emb(pos)

def decision_sequence(ds, idx, pred):
    """
    the ids `decision` pooling reads: the prompt plus the prefix of the
    model's output that copies the lemma. the decision step is its last
    position
    """
    return ds.prompt(idx) + ds.vocab.encode(copied(ds.srcs[idx], pred),
                                            bos=False, eos=False)

def window(a, end, k):
    """
    the k rows of `a` ending at position `end`, right-aligned and zero-padded
    on the left, flattened: templates of different-length verbs made
    comparable by lining up their ends
    """
    rows = a[max(0, end - k + 1):end + 1]
    pad = a.new_zeros(k - rows.size(0), *a.shape[1:])
    return torch.cat([pad, rows]).flatten()

@torch.enable_grad()
def decompose(lm, ids, k=6):
    """
    the decision-step analysis of one sequence: ReLU pattern, the model's
    chosen next token and its runner-up, templates for the chosen logit and
    for the chosen-minus-runner-up margin, per-position contributions, and
    the checks that the frozen map is exact
    """
    x0 = embed(lm, ids).detach().requires_grad_(True)
    store = {}
    logits = forward(lm, x0, store=store)
    t = len(ids) - 1
    top2 = logits[t].topk(2).indices
    chosen, runner = top2[0].item(), top2[1].item()

    a_chosen, = torch.autograd.grad(logits[t, chosen], x0, retain_graph=True)
    a_margin, = torch.autograd.grad(logits[t, chosen] - logits[t, runner], x0)
    with torch.no_grad():
        at_zero = forward(lm, torch.zeros_like(x0), frozen=store)[t]
        b_chosen = at_zero[chosen]
        linear_chosen = (a_chosen * x0).sum() + b_chosen
        # the frozen map must be affine: scaling the input moves the logit
        # exactly along the template
        scaled = forward(lm, 1.5 * x0, frozen=store)[t, chosen]
        linear_scaled = (a_chosen * 1.5 * x0).sum() + b_chosen
    # the templates of every candidate token at once: the Jacobian of all
    # logits at step t through the frozen map, (T, V, d). unlike a_chosen it
    # does not depend on which token the model chose, so similarity in it is
    # not similarity of the choice by construction
    jacobian = jacrev(lambda x: forward(lm, x, frozen=store)[t])(x0.detach())
    jacobian = jacobian.permute(1, 0, 2).detach()

    return {
        'code': torch.cat([store['relu_0'][t], store['relu_1'][t]]).bool(),
        'full_code': torch.cat([store['relu_0'], store['relu_1']], dim=-1).bool(),
        'chosen': chosen, 'runner': runner,
        'margin': (logits[t, chosen] - logits[t, runner]).item(),
        'a_chosen': a_chosen, 'a_margin': a_margin, 'jacobian': jacobian,
        'contrib_margin': (a_margin * x0).sum(-1).detach(),
        'exact_error': abs(linear_chosen.item() - logits[t, chosen].item()),
        'affine_error': abs(scaled.item() - linear_scaled.item()),
        't': t,
    }

def decompose_row(lm, ds, idx, pred, k=6):
    """
    `decompose` for row `idx` of `ds` with the model output `pred`, its two
    templates cut down to right-aligned windows so that verbs of different
    lengths are comparable: the k positions ending at the decision step (the
    copied stem) and the k ending at the last lemma segment before <sep>
    """
    ids = torch.tensor(decision_sequence(ds, idx, pred), device=lm.seg_emb.weight.device)
    item = decompose(lm, ids, k)
    sep = ids.tolist().index(ds.vocab.sep_id)
    for name in ('a_chosen', 'a_margin', 'jacobian'):
        a = item.pop(name)
        item[f'{name}_win'] = torch.cat([window(a, item['t'], k),
                                         window(a, sep - 1, k)]).detach().cpu()
    item['code'] = item['code'].cpu()
    return item

def decision_features(model, ds, preds, k=6) -> dict:
    """
    the decision-step features of every row of `ds` that the analogy methods
    compare: ReLU codes (N, 512); the margin and chosen-logit template
    windows (N, 2 * k * d_model); and the all-logit Jacobian windows
    (N, 2 * k * vocab * d_model), the choice-free template
    """
    model.lm.eval()
    rows = [decompose_row(model.lm, ds, idx, preds[idx], k) for idx in range(len(ds))]
    return {'code': torch.stack([r['code'] for r in rows]),
            'template': torch.stack([r['a_margin_win'] for r in rows]),
            'template_chosen': torch.stack([r['a_chosen_win'] for r in rows]),
            'jacobian': torch.stack([r['jacobian_win'] for r in rows])}

def analyze(model, splits, preds, k=6):
    """
    decomposes the decision step of every train and held-out row. returns a
    list of per-verb dicts with the fields `decompose_row` gives plus the
    verb, its output and the change class of that output
    """
    model.lm.eval()
    items = []
    for split in ('train', 'ftune', 'dev', 'test'):
        ds = splits[split]
        for idx in range(len(ds)):
            item = decompose_row(model.lm, ds, idx, preds[split][idx], k)
            item.update(split=split, src=ds.srcs[idx], pred=preds[split][idx],
                        cls=analogy.classify(ds.srcs[idx], preds[split][idx]))
            items.append(item)
    return items

def hamming(a, b):
    """
    pairwise Hamming distances between the rows of two boolean matrices
    """
    a, b = a.float(), b.float()
    return a @ (1 - b).T + (1 - a) @ b.T

def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = torch.as_tensor(x, dtype=torch.float)
    return '  '.join(f'q{int(q * 100)} {torch.quantile(x, q).item():6.1f}' for q in qs)

def report(items, seed=0):
    """
    step 2: how finely do the decision-step ReLU patterns partition the verbs,
    and does that partition have structure
    """
    rng = random.Random(seed)
    train = [it for it in items if it['split'] == 'train']
    held = [it for it in items if it['split'] != 'train']
    print(f'{len(train)} train and {len(held)} held-out verbs')

    err = max(it['exact_error'] for it in items)
    aff = max(it['affine_error'] for it in items)
    print(f'\nexactness: max |sum<A,X0> + b - logit| = {err:.2e}; '
          f'max affine error under scaling = {aff:.2e}')

    codes = torch.stack([it['code'] for it in items])
    width = codes.size(1) // 2
    on = codes.float().mean(0)
    for name, sl in (('layer 1', slice(0, width)), ('layer 2', slice(width, None))):
        frac = on[sl]
        print(f'{name}: {int((frac == 0).sum())} units never on, '
              f'{int((frac == 1).sum())} always on, '
              f'{int(((frac > 0) & (frac < 1)).sum())} vary; '
              f'mean units on per verb {codes[:, sl].float().sum(1).mean():.1f}')

    print('\nexact sharing of codes')
    for name, sl in (('both layers', slice(None)), ('layer 1', slice(0, width)),
                     ('layer 2', slice(width, None))):
        keys = [tuple(c[sl].nonzero().flatten().tolist()) for c in codes]
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        shared = sum(1 for key in keys if counts[key] > 1)
        print(f'  {name:12s}: {len(counts)} distinct codes for {len(keys)} verbs; '
              f'{shared} verbs share theirs with another')
    full = {}
    for it in items:
        key = it['full_code'].flatten().nonzero().flatten().tolist(), it['full_code'].shape
        full.setdefault((tuple(key[0]), tuple(key[1])), 0)
        full[(tuple(key[0]), tuple(key[1]))] += 1
    print(f'  {"all positions":12s}: {len(full)} distinct codes for {len(items)} verbs')

    tr = torch.stack([it['code'] for it in train])
    he = torch.stack([it['code'] for it in held])
    print('\nHamming distance to nearest training verb vs to a random one')
    for name, sl in (('both layers', slice(None)), ('layer 1', slice(0, width)),
                     ('layer 2', slice(width, None))):
        d_held = hamming(he[:, sl], tr[:, sl])
        d_train = hamming(tr[:, sl], tr[:, sl])
        d_train.fill_diagonal_(float('inf'))
        rand = [d_held[i, rng.randrange(len(train))].item() for i in range(len(held))]
        print(f'  {name}')
        print(f'    held-out, nearest : {quantiles(d_held.min(1).values)}')
        print(f'    train, nearest    : {quantiles(d_train.min(1).values)}')
        print(f'    held-out, random  : {quantiles(rand)}')


def neighbours_report(model, items, splits, preds):
    """
    step 2, last part: nearest training verb by region code, by decision
    hidden state and by template, and how often each shares the item's output
    class and final segment
    """
    train = [it for it in items if it['split'] == 'train']
    held = [it for it in items if it['split'] != 'train']
    tr = torch.stack([it['code'] for it in train])
    he = torch.stack([it['code'] for it in held])
    d_code = hamming(he, tr)

    # decision hidden states, as in the analogy analysis
    reps = {}
    for split in ('train', 'ftune', 'dev', 'test'):
        reps[split] = F.normalize(model.represent(splits[split], 'decision', preds[split]), dim=-1)
    hid_tr = reps['train']
    hid_he = torch.cat([reps[s] for s in ('ftune', 'dev', 'test')])
    sim_hidden = hid_he @ hid_tr.T

    tm_tr = F.normalize(torch.stack([it['a_margin_win'] for it in train]), dim=-1)
    tm_he = F.normalize(torch.stack([it['a_margin_win'] for it in held]), dim=-1)
    sim_template = tm_he @ tm_tr.T

    # rank agreement between code distance and hidden-state similarity, per
    # held-out verb, averaged
    def spearman(x, y):
        rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
        rx, ry = rx - rx.mean(), ry - ry.mean()
        return (rx @ ry / (rx.norm() * ry.norm())).item()
    rho_hidden = sum(spearman(-d_code[i], sim_hidden[i]) for i in range(len(held))) / len(held)
    rho_template = sum(spearman(-d_code[i], sim_template[i]) for i in range(len(held))) / len(held)
    print(f'\nmean Spearman, code closeness vs hidden-state cosine : {rho_hidden:.3f}')
    print(f'mean Spearman, code closeness vs margin-template cosine: {rho_template:.3f}')

    rng = random.Random(0)
    nearest = {
        'code (Hamming)': d_code.argmin(1).tolist(),
        'hidden (= decision)': sim_hidden.argmax(1).tolist(),
        'margin template': sim_template.argmax(1).tolist(),
        'random': [rng.randrange(len(train)) for _ in held],
    }
    same = {a: sum(x == y for x, y in zip(nearest['code (Hamming)'], nearest[a])) / len(held)
            for a in nearest}
    print('share of held-out verbs whose code-nearest neighbour is also nearest by: '
          + ', '.join(f'{a} {v:.2f}' for a, v in same.items() if a != 'code (Hamming)'))

    print('\nnearest training verb shares the held-out verb\'s ...')
    print(f'  {"neighbour by":22s} {"output class":>13s} {"(irregular)":>12s} {"final seg":>10s}')
    irregular = [i for i, it in enumerate(held)
                 if it['cls'] not in analogy.REGULAR and it['cls'] != 'other']
    for name, nb in nearest.items():
        cls = sum(held[i]['cls'] == train[n]['cls'] for i, n in enumerate(nb)) / len(held)
        irr = sum(held[i]['cls'] == train[nb[i]]['cls'] for i in irregular) / max(len(irregular), 1)
        fin = sum(held[i]['src'][-1:] == train[n]['src'][-1:] for i, n in enumerate(nb)) / len(held)
        print(f'  {name:22s} {cls:13.3f} {irr:12.3f} {fin:10.3f}')
    print(f'  ({len(irregular)} held-out verbs with irregular output)')
    return nearest

def main():
    parser = argparse.ArgumentParser(
        description='spline-view analysis of one trained run: ReLU regions and '
                    'templates at the decision step')
    parser.add_argument('--run', default='../results/en_0_1000')
    parser.add_argument('--k', type=int, default=6,
                        help='positions per right-aligned template window')
    parser.add_argument('--save', default=None,
                        help='where to torch.save the per-verb analysis')
    args = parser.parse_args()

    import run
    model, splits, _ = run.load_run(args.run)
    preds = {split: model.predict(ds) for split, ds in splits.items()}
    items = analyze(model, splits, preds, k=args.k)
    report(items)
    neighbours_report(model, items, splits, preds)
    if args.save:
        torch.save(items, args.save)

if __name__ == '__main__':
    main()
