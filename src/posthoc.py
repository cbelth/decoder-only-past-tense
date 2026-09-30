import argparse
import csv
import glob
import hashlib
import json
import os
import re
import time

import torch

import analogy
import run

def runs(results, seeds=None, sizes=None):
    """
    the (seed, size, directory) of every trained run under `results`,
    smallest size first
    """
    found = []
    for path in glob.glob(f'{results}/en_*_*/model.pt'):
        directory = os.path.dirname(path)
        match = re.fullmatch(r'en_(\d+)_(\d+)', os.path.basename(directory))
        if not match:
            continue
        seed, size = int(match.group(1)), int(match.group(2))
        if (seeds is None or seed in seeds) and (sizes is None or size in sizes):
            found.append((seed, size, directory))
    return sorted(found, key=lambda r: (r[1], r[0]))

def fingerprint(directory) -> str:
    """
    a hash of the run's model.pt, stored with its analogy results so that
    results from a model since retrained can be told apart. file times cannot
    do this: copying runs between machines resets them
    """
    with open(f'{directory}/model.pt', 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def is_current(directory, tag) -> bool:
    """
    whether the run has analogy results for `tag` computed from its current
    model.pt. results written before fingerprints were stored carry none and
    are trusted
    """
    out = f'{directory}/analogy.{tag}.json'
    if not os.path.exists(out):
        return False
    with open(out) as f:
        stored = json.load(f).get('model_sha1')
    return stored is None or stored == fingerprint(directory)

def stop_criterion(directory) -> str:
    """
    the --stop the run was trained with; runs from before the flag used loss
    """
    saved = torch.load(f'{directory}/model.pt', map_location='cpu', weights_only=False)
    return saved['args'].get('stop', 'loss')

def main():
    parser = argparse.ArgumentParser(
        description='reload every trained model under --results and rerun the '
                    'k=1 analogy analysis with the chosen methods, without '
                    'retraining. per run it writes {split}.analogy.<tag>.tsv '
                    'and analogy.<tag>.json; across runs, analogy.<tag>.csv and '
                    'analogy_confusion.<tag>.csv')
    parser.add_argument('--results', default='../results')
    parser.add_argument('--data', default=None,
                        help='data directory, if not the one stored at training '
                             'time (e.g. the runs were trained on another machine)')
    parser.add_argument('--methods', nargs='*', default=list(analogy.METHODS),
                        choices=analogy.METHODS)
    parser.add_argument('--tag', default='posthoc',
                        help='names the output files, so runs with different '
                             'method sets do not overwrite each other')
    parser.add_argument('--seeds', type=int, nargs='*')
    parser.add_argument('--sizes', type=int, nargs='*')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--redo', action='store_true',
                        help='recompute runs whose analogy.<tag>.json already '
                             'matches their model.pt')
    args = parser.parse_args()

    todo = runs(args.results, args.seeds, args.sizes)
    print(f'{len(todo)} trained runs under {args.results}, methods {args.methods}',
          flush=True)

    for i, (seed, size, directory) in enumerate(todo, 1):
        out = f'{directory}/analogy.{args.tag}.json'
        if is_current(directory, args.tag) and not args.redo:
            print(f'[{i}/{len(todo)}] seed {seed} size {size}: done, skipping', flush=True)
            continue
        start = time.time()
        model, splits, saved = run.load_run(directory, data=args.data,
                                            device=args.device)
        # the model's own outputs, from the model just loaded, so that
        # model_decision reads states along exactly what it produced
        preds = {split: model.predict(ds) for split, ds in splits.items()}
        summaries = run.analyze_analogy(model, splits, preds, directory,
                                        methods=args.methods, seed=saved.seed,
                                        suffix=f'.{args.tag}')
        with open(out, 'w') as f:
            json.dump({'seed': seed, 'size': size, 'methods': args.methods,
                       'model_sha1': fingerprint(directory),
                       'analogy': summaries}, f, indent=2)
        test = summaries['test']
        print(f'[{i}/{len(todo)}] seed {seed} size {size}: test class acc '
              + ', '.join(f'{m} {s["all"]["class_acc_pred"]:.3f}' for m, s in test.items())
              + f' ({time.time() - start:.0f}s)', flush=True)

    write_tables(args.results, args.tag)

RATES = ('applies', 'matches_pred', 'matches_gold', 'class_acc_pred',
         'class_acc_gold', 'kappa_pred')

def write_tables(results, tag) -> None:
    """
    long-format tables over every run with current results for `tag`, so learning
    curves are one groupby away:
    analogy.<tag>.csv, one row per seed x size x split x method x subset;
    analogy_confusion.<tag>.csv, one row per nonzero confusion cell, the class
    of the model's prediction against the class of the neighbour's output.
    both carry the run's stopping criterion, since a grid can mix runs trained
    under different ones, and leave out runs whose results are stale
    """
    summary = f'{results}/analogy.{tag}.csv'
    confusion = f'{results}/analogy_confusion.{tag}.csv'
    with open(summary, 'w', newline='') as f, open(confusion, 'w', newline='') as g:
        rates, cells = csv.writer(f), csv.writer(g)
        rates.writerow(['seed', 'size', 'stop', 'split', 'method', 'subset', 'n',
                        *RATES])
        cells.writerow(['seed', 'size', 'stop', 'split', 'method', 'class_pred',
                        'class_neighbor', 'count'])
        skipped = []
        for seed, size, directory in runs(results):
            if not is_current(directory, tag):
                skipped.append(os.path.basename(directory))
                continue
            stop = stop_criterion(directory)
            with open(f'{directory}/analogy.{tag}.json') as h:
                summaries = json.load(h)['analogy']
            for split, by_method in summaries.items():
                for method, summ in by_method.items():
                    for subset, r in summ.items():
                        if subset in ('per_class', 'confusion'):
                            continue
                        rates.writerow([seed, size, stop, split, method, subset,
                                        r['n'], *(r[k] for k in RATES)])
                    for truth, row in summ['confusion'].items():
                        for guess, count in row.items():
                            cells.writerow([seed, size, stop, split, method, truth,
                                            guess, count])
    print(f'wrote {summary} and {confusion}', flush=True)
    if skipped:
        print(f'left out {len(skipped)} runs with missing or stale results: '
              f'{", ".join(skipped)}', flush=True)

if __name__ == '__main__':
    main()
