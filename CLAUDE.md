# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# decoder-only past tense

A decoder-only (causal) transformer trained to inflect English verbs for the
past tense. The research question is whether the model behaves
**analogically**: can a test verb's output be predicted by taking its nearest
training verb in the model's hidden space and applying that verb's
lemma → past change? This includes irregulars.

## Layout

```
data/     en_{seed}_{size}.{train,ftune}, en_{seed}.{dev,test,gold,incrementaltrain}
src/      all code; run scripts from inside src/ (paths default to ../data, ../results)
results/  one directory per run, en_{seed}_{size}/ (model.pt is tracked)
```

Only seeds 0–9 are kept. Sizes are 100–1500 in steps of 100. That gives 150
(seed, size) runs.

## Data format

TSV: `source<TAB>target<TAB>features`. Segments are space-separated IPA, e.g.
`l a ɪ v	l a ɪ v d	V;PST`.

- `en_{seed}_{size}.train` / `.ftune`: an 80/20 split of the first `size`
  verbs of `en_{seed}.incrementaltrain`.
- `en_{seed}.dev`: 300 items.
- `en_{seed}.test` has **no target column**. `en_{seed}.gold` is the same
  rows with targets, so the code reads `.gold` as the test split.
- Irregulars are sparse. For seed 0 there are 102/1200 in train (size 1500),
  22/300 in ftune, 6/300 in dev and 7/300 in gold.

## Code (src/)

- `SegVocab.py`, `SegStrLM.py`: copied unchanged from `../ndp-adp/src/`.
  Keep them in sync with that copy instead of editing them here.
  `SegStrLM` is a small causal transformer with learned absolute positions.
- `SegStrPairDataset.py`: encodes a row as
  `<bos> <V> <PST> source <sep> target <eos>`. Feature tags are one token each.
- `SegStrTransducer.py`: a Lightning module wrapping `SegStrLM`, with the loss
  on target tokens only. `--source_weight` (default 0) adds a weighted LM
  loss over the prompt too. It adds:
  - `predict` / `evaluate`: greedy decoding and exact-match word accuracy.
    Prompts are grouped by length because `generate` can't take padded
    prompts.
  - `represent(ds, pool, preds=None)`: one vector per lemma. `pool` is one
    of:
    - `sep`: the state at `<sep>`.
    - `mean`: the average over the prompt.
    - `stem`: appends an unchanged copy of the lemma after `<sep>` and reads
      the last state, i.e. where the suffix is decided.
    - `decision`: feeds in the model's own output (`preds`) for as long as
      it copies the lemma, and reads the state at the first step where it
      departs (`copied()` gives that prefix). This is the state that chose
      the suffix, the changed vowel or `<eos>`. It is identical to `stem`
      whenever the output is regular. No output token is ever in its
      input, so the answer never leaks into the vector. Pooling over all
      output steps was rejected because later steps have the output in
      context, which would make the analogy test circular.
  - `nearest(query, ref, k, pool, query_preds, ref_preds)`: cosine kNN.
    Train rows use the model's own predictions too, not gold.
- `analogy.py`:
  - `change(lemma, past)` strips the shared prefix and suffix to get an
    end-anchored rewrite `old → new / _ right#`.
  - `apply` performs that rewrite, or returns None when it doesn't apply.
  - `classify(lemma, past)` assigns one of `CLASSES`, our own scheme (there
    may be an established one to switch to):

    | class | example |
    |---|---|
    | `reg_d`, `reg_t`, `reg_əd` | `l a ɪ v → l a ɪ v d` |
    | `no_change` | `h ɪ t → h ɪ t` (the sing → sing type) |
    | `vowel_change` | `s ɪ ŋ → s æ ŋ`, `t e ɪ k → t ʊ k` |
    | `vowel_change_suffix` | `k iː p → k ɛ p t`, `t ɛ l → t o ʊ l d` |
    | `rhyme_replaced` | `θ ɪ ŋ k → θ ɔ t`, `b r ɪ ŋ → b r ɔ t` |
    | `coda_change` | `b ɪ l d → b ɪ l t`, `m e ɪ k → m e ɪ d` |
    | `other` | `ɡ o ʊ → w ɛ n t`, and most malformed model outputs |

    Irregulars are split on the final syllable (`final_syllable`: last run
    of `VOWELS` as the nucleus, so `e ɪ` is one nucleus). Did the nucleus
    change, and did the coda stay the same, gain `t`/`d`, or get replaced by
    material ending in `t`/`d`? Any change before the final nucleus is
    `other`. Known compromises: `buy → bought` is `vowel_change_suffix`,
    `fight → fought` is `vowel_change`, and `stand → stood` and
    `hear → heard` are `rhyme_replaced`.
  - `METHODS` are `model_{sep,mean,stem,decision}` plus these baselines:
    - `string`: nearest by edit distance, ties broken at random.
    - `final`: a random training verb with the same last segment. This is
      the one to beat on regulars, since the last segment decides the
      allomorph.
    - `majority`: always the most common class.
    - `random`: a random training verb.
  - `neighbor_sets` / `run_methods` compute the k=1 neighbours and analogy
    rows for any subset of these. `write_tsv` writes the analogy table.
  - `analogize` scores each item two ways. The neighbour's past tense is
    **the model's own output for that training verb** (`ref_preds`), not
    gold.
    - Exact: apply the neighbour's `change` to the test lemma and compare
      the resulting string. A change that doesn't apply counts as a miss,
      so `applies` is reported too.
    - Class: the neighbour's class vs. the class of test lemma → model
      output (`class_acc_pred`) and test lemma → gold (`class_acc_gold`).
      This is the main measure.
  - `summarize` gives rates, plus Cohen's kappa for class agreement, for
    these subsets: `all`, `gold_regular` / `gold_irregular`, and
    `pred_regular` / `pred_irregular` / `pred_other`, split by the class of
    the model's output. `other` is kept out of `pred_irregular` because it
    is mostly malformed. It also gives per-class recall and a confusion
    matrix (rows: class of the model's output; columns: the neighbour's).
  - Accuracy is dominated by the three regular classes. Read kappa and
    `pred_irregular` against the `final` and `majority` baselines.
- `run.py`: one (seed, size) run.
  - `--seed` picks the data split and also seeds training
    (`seed_everything`, `deterministic=True`).
  - **The vocab and max_len are built over all splits.** Small train files
    lack segments like `ð d͡ʒ t͡ʃ`, and eval forms are longer than train
    forms.
  - Training validates on **ftune**. `--stop` picks both when to stop and
    which epoch to keep (`StopAndSelect`), with `--patience 20` and
    `--epochs 500` max:
    - `loss_train_irregular` (**default**): patience resets on a new low in
      ftune loss or a new high in train irregular accuracy. It keeps the
      lowest-ftune-loss epoch among those with train irregular accuracy ≥
      `--train_irr_target` (0.9), or the highest train irregular accuracy
      if none reaches it.
    - `loss`: what the server runs used. Stops on ftune loss and keeps the
      lowest-loss epoch.
    - `loss_irregular`: uses ftune (held-out) irregular accuracy instead of
      train. Not useful: about 22 held-out irregulars, of which the model
      gets 0–2.
  - Each epoch logs `ftune_irr_acc` and `train_irr_acc` (word accuracy on
    the irregular rows) in `metrics.csv`.
  - Why the default changed: with `loss`, train irregular accuracy averaged
    0.39 at size 1500 (regulars 0.995), because ftune loss flattens while
    irregulars are still being learned. Seed 0:

    | size | stop | epoch kept | train irregular | train | dev | test |
    |---|---|---|---|---|---|---|
    | 1500 | `loss` | 91 of 112 | ≈0.63 | 0.970 | 0.873 | 0.873 |
    | 1500 | `loss_train_irregular` | 145 of 208 | 0.902 | 0.990 | 0.880 | 0.893 |
    | 300 | `loss` | 70 of 91 | 0.759 | 0.938 | 0.223 | 0.273 |
    | 300 | `loss_train_irregular` | 88 of 123 | 0.944 | 0.988 | 0.247 | 0.290 |

    The server results in `results/` were trained with `loss`. Retrain
    into a new `--out` (or with `--redo`) before comparing.
  - It then writes the files listed below and runs the analogy analysis
    (`analyze_analogy`, which `posthoc.py` reuses).
    `--pool` only picks the pooling for `neighbors.tsv`. `analogy.tsv` always
    covers all three pools.
  - `load_run(dir, data=None, device='cpu')` rebuilds a trained model and its
    splits for post-hoc work.
- `sweep.py`: loops over every (seed, size) in `data/`, smallest size first.
  - Skips runs that already have `scores.json`, and restarts interrupted runs
    from scratch.
  - Options it doesn't know are passed through to `run.py`.
  - `--seeds` / `--sizes` restrict the grid.
- `posthoc.py`: reloads every `results/en_*_*/model.pt` and reruns the
  analogy analysis without retraining. It recomputes predictions from the
  loaded model.
  - `--methods` picks a subset of `analogy.METHODS`.
  - `--tag` (default `posthoc`) names the outputs, so different method sets
    don't overwrite each other.
  - It skips runs that already have `analogy.<tag>.json` (`--redo` to
    recompute).
  - `--data` overrides the stored data path.
  - It then writes two long-format tables across all runs:
    - `results/analogy.<tag>.csv`: seed, size, split, method, subset, n,
      applies, matches_pred, matches_gold, class_acc_pred, class_acc_gold,
      kappa_pred.
    - `results/analogy_confusion.<tag>.csv`: seed, size, split, method,
      class_pred, class_neighbor, count.

## Running

The only dependencies are `torch` and `lightning` (developed on torch 2.13,
lightning 2.6). There is no requirements file, test suite or linter. To
check a change, do a quick run such as
`python run.py --seed 0 --size 100 --epochs 5 --out /tmp/check`.

```
cd src
python run.py --seed 0 --size 1500            # one run
python sweep.py                               # full grid, resumable
python sweep.py --seeds 0 1 2 --sizes 100 200 # subset
nohup python sweep.py > ../sweep.log 2>&1 &   # on the server
python posthoc.py                             # analogy over all saved models
python posthoc.py --methods model_decision string random --tag decision
```

The model is tiny (about 100k params, 64-dim), so a GPU speeds up each run
only a little. For throughput, run several sweeps in parallel over disjoint
`--seeds`. On a laptop (MPS), size 1500 took about 30s to early-stop with
patience 5.

Post-hoc:

```python
import run
model, splits, args = run.load_run('../results/en_0_1500', device='cuda')
model.evaluate(splits['dev'])['acc']
model.nearest(splits['test'], splits['train'], k=1, pool='stem')
```

## Per-run outputs (results/en_{seed}_{size}/)

- `model.pt`: weights plus args and LM hparams. `vocab.json`.
- `scores.json`: word accuracy per split, best epoch, and analogy summaries.
  It is written last, so it marks a finished run.
- `{split}.preds.tsv`: source, gold, prediction, correct.
- `{split}.neighbors.tsv`: top-k neighbours under `--pool`.
- `{split}.analogy.tsv`: one row per item per method, written at training
  time by the code of that moment (the server runs predate `decision` and
  the classes).
- `{split}.analogy.posthoc.tsv`: the current analysis. It adds
  `neighbor_out` (the model's output for the neighbour), `neighbor_gold`,
  `class_gold`, `class_pred` and `class_neighbor`.
- `{split}.analogy.<tag>.tsv`, `analogy.<tag>.json`: the same, from
  `posthoc.py`.
- `version_0/metrics.csv`: per-epoch train/ftune loss and token accuracy.
  The `val_*` columns are ftune.

ftune picks the checkpoint, so **dev and test are the clean held-out scores**.

## Findings so far (seed 0, size 1500, before the early-stopping version)

These runs are not in `results/`, which starts empty.

- Word accuracy was about 0.84 on dev and test.
- k=1 analogy matching the model's output on test:

  | neighbour | match |
  |---|---|
  | `stem` | 0.843 |
  | `string` | 0.667 |
  | `sep` | 0.430 |
  | `mean` | 0.390 |
  | `random` | 0.283 |

- `sep` neighbours always share the lemma's **first** segment, and all
  cosines are about 0.99. The `<sep>` state predicts the first output
  segment, which is a copy of the lemma's first segment, so it doesn't
  capture what the past tense depends on.
- Caveat on `stem`: it reads the state that predicts the suffix, so its
  nearest neighbours will largely share the model's chosen allomorph by
  construction. Its high match rate on regulars is close to guaranteed. The
  informative cases are irregular/vowel-change outputs.
- All 7 test irregulars were regularized by the model, so the
  "irregular" column is not yet a test of analogy.
- `decision` (reloaded from the same kind of run) matched 0.840 on test,
  about the same as `stem`, which is expected since they coincide on regular
  outputs. On ftune, 29/300 outputs were not regular. Most of those were
  malformed, with only a few real irregulars. For those, `decision`'s
  neighbours look analogical even when the strict rewrite fails:
  - `f ɹ̩ ɡ ɪ v → f ɹ̩ ɡ e ɪ d` gets neighbour `ɡ ɪ v → ɡ e ɪ v`
  - `o ʊ v ɹ̩ k ʊ k → o ʊ v ɹ̩ k e ɪ k` gets neighbour `o ʊ v ɹ̩ k ə m →
    o ʊ v ɹ̩ k e ɪ m`
  
  Exact rule application may be too strict a criterion for irregulars.

## Findings: class agreement over the full sweep (posthoc, 150 runs)

From `results/analogy.posthoc.csv` and `analogy_confusion.posthoc.csv`.

- **Test class agreement with the model's output**, mean over 10 seeds:
  - `decision`: 0.14 at size 100, 0.49 at 500, 0.89 at 1500
  - `stem`: 0.86 at 1500
  - `final`: 0.83 at 1500
  - `string`: 0.67 at 1500
  - `majority`: 0.47 at 1500
  - `random`: 0.31 at 1500
  - Kappa at 1500: `decision` 0.83, `final` 0.75.
- **Irregular outputs** (`pred_irregular`, ftune+dev+test pooled over
  seeds): `decision` 0.33–0.47 across sizes, `stem` 0.17–0.34, `string`
  about 0.2, `final` 0.05–0.15, `random` about 0.02.
- **Per-class recall** (sizes 1000–1500, `decision`):
  - `no_change` 0.66, `vowel_change` 0.74
  - `vowel_change_suffix` 0.24, `coda_change` 0.09, `rhyme_replaced` 0.04
  
  `decision` reads the state that chose the **first** departing segment.
  That captures `ɪ → æ` but not what happens to the coda afterwards.
- **Caveat:** `decision` is close to perfect on regulars (0.98–0.995)
  largely by construction. The state it reads produces the next segment
  (`d`/`t`/`ə`), so close states choose the same suffix. The same applies,
  more weakly, to the first vowel of an irregular.
- **Small sizes are mostly malformed output.** The share of test outputs
  classed `other` falls from 0.83 at size 100 and 0.51 at 500 to 0.11 at
  1500. Train accuracy is about 0.9 even at size 100, while test is about
  0.02.

## Open next steps

- Look for an established classification of English past-tense
  irregulars, to replace or validate `CLASSES`.
- Pool irregulars across ftune, dev and test, and across seeds and sizes.
- Plot learning curves from `results/analogy.posthoc.csv` (class agreement
  and kappa vs. size, per method).
