import argparse
import glob
import os
import re
import shutil
import sys
import time
import traceback

import run

def available(data):
    """
    the (seed, size) pairs with a .train file in `data`
    """
    pairs = set()
    for path in glob.glob(f'{data}/en_*_*.train'):
        match = re.fullmatch(r'en_(\d+)_(\d+)\.train', os.path.basename(path))
        if match:
            pairs.add((int(match.group(1)), int(match.group(2))))
    return sorted(pairs)

def main():
    parser = argparse.ArgumentParser(
        description='train and analyse one model per (seed, size) with run.py '
                    'defaults. any option this script does not know is passed '
                    'through to run.py, e.g. --patience 30')
    parser.add_argument('--data', default='../data')
    parser.add_argument('--out', default='../results')
    parser.add_argument('--seeds', type=int, nargs='*',
                        help='default: every seed in --data')
    parser.add_argument('--sizes', type=int, nargs='*',
                        help='default: every size in --data')
    parser.add_argument('--redo', action='store_true',
                        help='retrain runs that already finished')
    args, passthrough = parser.parse_known_args()

    jobs = [(seed, size) for seed, size in available(args.data)
            if (args.seeds is None or seed in args.seeds)
            and (args.sizes is None or size in args.sizes)]
    # smallest first, so the whole grid fills in at low sizes before the
    # slow large-size runs
    jobs.sort(key=lambda job: (job[1], job[0]))
    print(f'{len(jobs)} runs: seeds {sorted({s for s, _ in jobs})}, '
          f'sizes {sorted({n for _, n in jobs})}', flush=True)

    failed = []
    for i, (seed, size) in enumerate(jobs, 1):
        out = run.run_dir(args.out, seed, size)
        if os.path.exists(f'{out}/scores.json') and not args.redo:
            print(f'[{i}/{len(jobs)}] seed {seed} size {size}: done, skipping',
                  flush=True)
            continue
        # a run without scores.json was interrupted; start it clean rather
        # than leave a stale log version or checkpoint beside the new one
        shutil.rmtree(out, ignore_errors=True)

        print(f'[{i}/{len(jobs)}] seed {seed} size {size}', flush=True)
        start = time.time()
        try:
            run.main(['--seed', str(seed), '--size', str(size), '--data', args.data,
                      '--out', args.out, '--quiet', *passthrough])
        except Exception:
            # one bad run should not cost the rest of the grid
            traceback.print_exc()
            failed.append((seed, size))
        print(f'  {time.time() - start:.0f}s', flush=True)

    if failed:
        print(f'{len(failed)} failed: {failed}', flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
