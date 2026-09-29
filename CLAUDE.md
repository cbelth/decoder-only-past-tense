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
results/  one directory per run, en_{seed}_{size}/ (model.pt is gitignored)
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
  - `represent(ds, pool)`: one vector per lemma, built from the prompt only.
    `pool` is `sep` (state at `<sep>`), `mean` (average over the prompt) or
    `stem` (appends an unchanged copy of the lemma after `<sep>` and reads
    the last state, i.e. where the suffix is decided).
  - `nearest`: cosine kNN between two datasets.
- `analogy.py`:
  - `change(lemma, past)` strips the shared prefix and suffix to get an
    end-anchored rewrite `old → new / _ right#`.
  - `apply` performs that rewrite, or returns None when it doesn't apply.
  - `neighbor_sets` gives k=1 neighbours under each pool plus two baselines:
    string edit distance and random.
  - `analogize` / `summarize` compute the match rate against both the model's
    prediction and the gold form, split by whether the gold form is regular.
    A rule that doesn't apply counts as a miss, so `applies` is reported
    too. `is_regular` means lemma + `d`, `t` or `ə d`, without checking that
    it is the right allomorph.
- `run.py`: one (seed, size) run.
  - `--seed` picks the data split and also seeds training
    (`seed_everything`, `deterministic=True`).
  - **The vocab and max_len are built over all splits.** Small train files
    lack segments like `ð d͡ʒ t͡ʃ`, and eval forms are longer than train
    forms.
  - Training validates on **ftune**, stops early on ftune loss
    (`--patience 20`, `--epochs 500` max), and restores the best epoch.
  - It then writes the files listed below and runs the analogy analysis.
    `--pool` only picks the pooling for `neighbors.tsv`. `analogy.tsv` always
    covers all three pools.
  - `load_run(dir, data=None, device='cpu')` rebuilds a trained model and its
    splits for post-hoc work.
- `sweep.py`: loops over every (seed, size) in `data/`, smallest size first.
  - Skips runs that already have `scores.json`, and restarts interrupted runs
    from scratch.
  - Options it doesn't know are passed through to `run.py`.
  - `--seeds` / `--sizes` restrict the grid.

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

- `model.pt`: weights plus args and LM hparams (gitignored). `vocab.json`.
- `scores.json`: word accuracy per split, best epoch, and analogy summaries.
  It is written last, so it marks a finished run.
- `{split}.preds.tsv`: source, gold, prediction, correct.
- `{split}.neighbors.tsv`: top-k neighbours under `--pool`.
- `{split}.analogy.tsv`: one row per item per neighbour type (`model_sep`,
  `model_mean`, `model_stem`, `string`, `random`).
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

## Open next steps

- Split the analogy match rate by the **model's** output type (regular /
  irregular / malformed) rather than by gold.
- Pool irregulars across ftune, dev and test, and across seeds and sizes.
  Smaller sizes should give more irregular outputs and errors.
- Aggregate `results/*/scores.json` into learning curves (accuracy and
  analogy match vs. size).
