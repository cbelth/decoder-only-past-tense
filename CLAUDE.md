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
  - `METHODS` are `model_{sep,mean,stem,decision}`, the spline methods
    `model_region`, `model_template` and `model_template_chosen` (see
    `spline.py`), plus these baselines:
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
- `spline.py`: the spline view of the decision step, motivated by
  Balestriero & Baraniuk 2018, "A Spline Theory of Deep Networks".
  - `forward` is `SegStrLM`'s forward pass written by hand (verified against
    `lm.forward` to about 1e-5). In storing mode it detaches and saves the
    attention weights, LayerNorm scales and ReLU masks. In frozen mode it
    reuses them, which makes the network exactly affine in the input
    embeddings X₀: logit_c = Σ_s ⟨A_c[s], X₀[s]⟩ + b_c.
  - `decompose` runs this on the `decision` sequence (prompt plus the copied
    stem) at the decision step. It returns:
    - the ReLU code: 256 + 256 bits at that position (the region);
    - templates, as gradients through the frozen map, for the chosen logit
      and for the chosen-minus-runner-up margin;
    - per-position contributions;
    - exactness checks: Σ⟨A,X₀⟩ + b = logit, and linearity under scaling.
  - Templates are compared through two right-aligned windows of `k=6`
    positions: one ending at the decision step (copied stem), one ending
    before `<sep>` (the lemma).
  - `decision_features` gives these for a dataset. The analogy methods use
    them:
    - `model_region`: least Hamming distance between codes, ties random.
    - `model_template`: cosine between margin templates.
    - `model_template_chosen`: cosine between chosen-logit templates.
  - `python spline.py --run ../results/en_0_1000` prints the region report:
    live units, exact code sharing, nearest-vs-random Hamming distances,
    agreement with the hidden state, and neighbour class and final-segment
    match.
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
  - `analogy.<tag>.json` stores a SHA-1 of the `model.pt` it was computed
    from. A run is skipped only if that matches, so a retrained model is
    always reanalysed (`--redo` recomputes anyway). File times are not used,
    because copying runs between machines resets them. Results from before
    the hash was stored have none and are trusted.
  - `--data` overrides the stored data path.
  - It then writes two long-format tables across all runs:
    - `results/analogy.<tag>.csv`: seed, size, stop, split, method, subset, n,
      applies, matches_pred, matches_gold, class_acc_pred, class_acc_gold,
      kappa_pred.
    - `results/analogy_confusion.<tag>.csv`: seed, size, stop, split,
      method, class_pred, class_neighbor, count.
    - `stop` is the run's training criterion, since a grid can mix them.
      Filter on it before comparing sizes. Runs with missing or stale
      results are left out, and the script lists them.

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

## Findings: spline view (seed 0, size 1000, `loss_train_irregular`)

- **Templates are exact** to about 1e-5.
- **Regions:** all 512 decision-step ReLUs vary across verbs. No two verbs
  share a full code; 260 of 1,600 share a layer-1-only code. Median Hamming
  distance to the nearest training verb is 39, against 214 for a random
  one, so regions are unique but strongly structured.
- **Redundancy:** region distance largely tracks the hidden state (Spearman
  0.73; same nearest neighbour 54% of the time). The margin template is a
  different view (0.39; 37%).
- **Irregular outputs** (43, ftune+dev+test), class agreement:
  - `model_template` and `model_template_chosen` 0.58
  - `decision` 0.47, `region` 0.44
  - `string` 0.16, `final` 0.09
- **The gain is the `hit` type.** For `t`/`d`-final verbs the model leaves
  unchanged, the hidden-state neighbour is often a regular `-əd` verb (both
  sit near the stop-vs-`ə d` boundary). The template neighbour is another
  unchanged verb, often phonologically unlike it: `r ɪ d` → `ʃ ʌ t`,
  `s k ɪ d` → `s p r ɛ d`. Template neighbours match the final segment
  less often (0.73 vs 0.83).
- `template_chosen` doing as well as `template` suggests the gain isn't an
  artefact of which runner-up the margin is taken against.
- **Region novelty:** of the 10% of held-out verbs farthest (Hamming) from
  every training region, 95% get malformed (`other`) output, against 16%
  overall.

## Findings: spline methods across the full retrained grid (150 runs, `loss_train_irregular`)

Class agreement on irregular outputs (ftune+dev+test, pooled over seeds):

| size | 100 | 300 | 500 | 900 | 1100 | 1500 |
|---|---|---|---|---|---|---|
| `template_chosen` | 0.38 | 0.52 | 0.53 | 0.53 | 0.58 | 0.65 |
| `template` (margin) | 0.30 | 0.47 | 0.50 | 0.50 | 0.55 | 0.64 |
| `decision` | 0.34 | 0.42 | 0.40 | 0.41 | 0.44 | 0.55 |
| `region` | 0.31 | 0.33 | 0.31 | 0.35 | 0.39 | 0.49 |
| `string` | 0.17 | 0.18 | 0.18 | 0.21 | 0.22 | 0.24 |
| `final` | 0.15 | 0.09 | 0.09 | 0.08 | 0.06 | 0.07 |

- `template_chosen` beats `decision` in 9–10 of 10 seeds at every size from
  200 up (7/10 at 100).
- The margin `template` is weaker at small sizes (2/10 seeds at 100, 5/10
  at 200) and catches up from about 500.
- On all test items, κ is nearly identical across model methods; regulars
  dominate.
- At 1500 the gain is concentrated in `no_change` (recall 0.67 → 0.94);
  `vowel_change` is about 0.9 for every model method.
- `vowel_change_suffix`, `rhyme_replaced` and `coda_change` are about 0.2
  for every method, since the decision step sees only the first change.

## Caveat on the template results (found after the grid run)

- **The chosen-logit template isn't choice-free.** It equals the Jacobian
  of the final hidden state times the output weight row of the chosen
  token, so verbs with the same chosen token share a fixed factor. The
  margin template likewise depends on the winner and the runner-up.
  Template similarity is therefore partly similarity of the model's
  choice, the same circularity as `stem`.
- **Check on seed 0 / 1500 (decision step, irregular outputs, n=43):** a
  choice-free template (Jacobian of *all* logits, 34,560 dims) gets 22/43,
  against 25/43 for the hidden state and 27/43 for `template_chosen`. On
  `no_change` it gets 13/16, against 14/16 and 16/16. The template's lead
  over the hidden state disappears once the choice is removed.
- **The hidden state isn't choice-free either:** logits = W·h. Hidden
  neighbours share the chosen token 95% of the time. Any representation
  read at the step where the choice is made encodes that choice, which
  limits what agreement scores at that step can show.
- **Resolved on the full grid.** `model_jacobian` (choice-free) was added
  and all 150 runs rerun. Class agreement on irregular outputs (pooled):

  | size | 100 | 300 | 500 | 900 | 1100 | 1500 |
  |---|---|---|---|---|---|---|
  | `decision` (hidden) | 0.34 | 0.42 | 0.40 | 0.41 | 0.44 | 0.55 |
  | `jacobian` (choice-free) | 0.35 | 0.36 | 0.33 | 0.35 | 0.42 | 0.48 |
  | `template_chosen` (leaky) | 0.38 | 0.52 | 0.53 | 0.53 | 0.58 | 0.65 |

  The Jacobian is below `decision` in 8–10 of 10 seeds at most sizes. **So
  the template advantage was the choice leak. Choice-free templates are no
  better than, and mostly slightly worse than, the hidden state as a basis
  for analogy.** What stands is that all model-based neighbours far exceed
  phonological ones (`string` about 0.2, `final` < 0.1 on irregular
  outputs).

## The walk (`walk.py`)

A held-out verb is generated one step at a time. At each step the nearest
step of any training verb is found, and its edit operation is applied to the
held-out lemma.
- **Operations** come from aligning each training verb's lemma with the
  model's own output for it (`align`: Levenshtein, backtraced with ties
  resolved match, then deletion, then substitution, then insertion, so
  deletions sit at the right edge). They are relative to a pointer into
  the lemma: COPY, SUB y, INS y and STOP, with skipped lemma segments
  attached as `skip`.
- **Steps are keyed by** `jacobian` (all-logit Jacobian at the step,
  windowed and count-sketched to 4096 dims; choice-free), `hidden` (the
  final-layer state), or `context` (symbolic baseline: last two output
  segments plus the next two lemma segments).
- **Scored by** free-walk fidelity to the model's output and to gold, and
  forced step agreement along the model's own path.
- `python walk.py --run ../results/en_0_1500` writes `<run>/walk.tsv`.
  It takes several minutes at size 1500 (each step needs a Jacobian).

Seed 0 / 1500, held-out (900 verbs):

| | `jacobian` | `hidden` | `context` |
|---|---|---|---|
| walk = model output, all | 0.877 | 0.886 | 0.694 |
| regular (802) | 0.963 | 0.960 | 0.772 |
| `no_change` (16) | 0.750 | 0.938 | 0.312 |
| other irregular (27) | 0.19 | 0.26 | 0 |
| walk = gold | 0.892 | 0.877 | 0.739 |
| forced SUB steps (114) | 0.04 | 0.06 | 0.01 |
| forced INS / STOP | 0.93 / 0.99 | 0.93 / 0.99 | 0.84 / 0.95 |

- **Substitutions almost never transfer.** At a vowel-change step the
  nearest training step is usually a COPY, so the walk keeps the vowel and
  adds a regular suffix (`f iː l → f iː l d`). The walk behaves as a
  regularizing analogical model: it reproduces suffix and stop decisions
  well and stem changes poorly.
- **At small sizes the walk can beat the model on gold** (seed 0 / 300:
  0.49 against the model's 0.29), because pointer-based copying can't
  garble the stem. That's the walk's structure, not the representation.

## Open next steps

- Look for an established classification of English past-tense
  irregulars, to replace or validate `CLASSES`.
- Pool irregulars across ftune, dev and test, and across seeds and sizes.
- Plot learning curves from `results/analogy.posthoc.csv` (class agreement
  and kappa vs. size, per method).
