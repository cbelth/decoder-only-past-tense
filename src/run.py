import argparse
import json
import os

# cuBLAS refuses deterministic mode on GPU unless this is set before CUDA
# initialises; harmless on CPU
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import analogy
from SegStrLM import SegStrLM
from SegStrPairDataset import SegStrPairDataset
from SegStrTransducer import SegStrTransducer
from SegVocab import SegVocab

def paths(data, seed, size):
    """
    the files for one seed and training size. the test split is read from
    .gold, since .test carries no targets and the two agree row for row
    """
    return {'train': f'{data}/en_{seed}_{size}.train',
            'ftune': f'{data}/en_{seed}_{size}.ftune',
            'dev': f'{data}/en_{seed}.dev',
            'test': f'{data}/en_{seed}.gold'}

def load(files):
    """
    reads every split against one vocab and one max_len. both are taken over
    all splits, not train alone: small train files lack segments the eval
    splits use, which would otherwise encode silently as <unk>, and eval forms
    run longer than any train form would size the positional table for. only
    the segment inventory and a length bound are shared; no targets are
    """
    raw = {split: SegStrPairDataset(path) for split, path in files.items()}
    vocab = SegVocab.from_sequences(
        [list(f) + s + t for ds in raw.values()
         for f, s, t in zip(ds.feats, ds.srcs, ds.tgts)])
    splits = {split: SegStrPairDataset(ds.path, vocab=vocab, rows=ds.rows)
              for split, ds in raw.items()}
    return splits, vocab, max(ds.max_len for ds in splits.values())

def build_parser():
    parser = argparse.ArgumentParser(
        description='train a SegStrTransducer on one seed and size of the '
                    'English past tense data, stopping early on ftune loss, '
                    'then score every split and run the analogy analysis')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--data', default='../data')
    parser.add_argument('--out', default='../results')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper bound; early stopping usually ends sooner')
    parser.add_argument('--patience', type=int, default=20,
                        help='epochs without a better ftune loss before stopping')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--source_weight', type=float, default=0.0)
    parser.add_argument('--pool', choices=['sep', 'mean', 'stem'], default='sep',
                        help='how a lemma\'s hidden states become one vector')
    parser.add_argument('--k', type=int, default=1,
                        help='nearest training lemmas to report per item')
    parser.add_argument('--quiet', action='store_true',
                        help='no progress bar or model summary, for batch runs')
    return parser

def run_dir(out, seed, size):
    return f'{out}/en_{seed}_{size}'

def build(args, splits, vocab, max_len):
    """
    the transducer `args` describe, untrained. ftune is its validation set,
    which is what early stopping and checkpoint selection watch
    """
    lm = SegStrLM(len(vocab), vocab.pad_id, d_model=args.d_model,
                  nhead=args.nhead, num_layers=args.num_layers,
                  dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                  max_len=max_len)
    return SegStrTransducer(lm, lr=args.lr, source_weight=args.source_weight,
                            train_ds=splits['train'], val_ds=splits['ftune'],
                            batch_size=args.batch_size)

def train(args):
    """
    trains until ftune loss stops improving for `patience` epochs, then
    restores the weights from the epoch where it was lowest and saves them
    with everything needed to rebuild the model (see `load_run`)
    """
    L.seed_everything(args.seed)
    splits, vocab, max_len = load(paths(args.data, args.seed, args.size))
    model = build(args, splits, vocab, max_len)

    run = run_dir(args.out, args.seed, args.size)
    os.makedirs(run, exist_ok=True)
    best = ModelCheckpoint(dirpath=run, filename='best', monitor='val_loss',
                           mode='min', save_top_k=1, save_weights_only=True,
                           enable_version_counter=False)
    stop = EarlyStopping(monitor='val_loss', mode='min', patience=args.patience)
    trainer = L.Trainer(max_epochs=args.epochs, logger=CSVLogger(run, name=''),
                        callbacks=[best, stop], deterministic=True,
                        enable_progress_bar=not args.quiet,
                        enable_model_summary=not args.quiet,
                        log_every_n_steps=1)
    trainer.fit(model)

    # the checkpoint also holds Lightning's bookkeeping; keep only lm.pt
    ckpt = torch.load(best.best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    os.remove(best.best_model_path)

    info = {'best_epoch': ckpt['epoch'], 'best_ftune_loss': best.best_model_score.item(),
            'epochs_run': trainer.current_epoch}
    torch.save({'args': vars(args), 'lm_hparams': dict(model.lm.hparams),
                'state_dict': model.state_dict(), **info}, f'{run}/model.pt')
    vocab.save(f'{run}/vocab.json')
    return model, splits, info

def load_run(run, data=None, device='cpu'):
    """
    rebuilds a trained model from a run directory, with its splits encoded
    against the vocab it was trained with. `data` overrides the data path
    stored at training time, for when the run has moved machines
    """
    saved = torch.load(f'{run}/model.pt', map_location=device, weights_only=False)
    args = argparse.Namespace(**saved['args'])
    if data is not None:
        args.data = data
    vocab = SegVocab.load(f'{run}/vocab.json')
    splits = {split: SegStrPairDataset(path, vocab=vocab)
              for split, path in paths(args.data, args.seed, args.size).items()}
    lm = SegStrLM(**saved['lm_hparams'])
    model = SegStrTransducer(lm, lr=args.lr, source_weight=args.source_weight,
                             train_ds=splits['train'], val_ds=splits['ftune'],
                             batch_size=args.batch_size)
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), splits, args

def analyze(model, splits, args, info=None):
    """
    scores every split by greedy decoding and runs the nearest-neighbour and
    analogy analyses, writing everything into the run directory. ftune picked
    the checkpoint, so dev and test are the clean held-out scores
    """
    run = run_dir(args.out, args.seed, args.size)
    scores, preds = {}, {}
    for split in ('train', 'ftune', 'dev', 'test'):
        ds = splits[split]
        result = model.evaluate(ds)
        scores[split] = result['acc']
        preds[split] = result['preds']
        with open(f'{run}/{split}.preds.tsv', 'w', encoding='utf-8') as f:
            for row, pred, ok in zip(ds.rows, result['preds'], result['correct']):
                f.write('\t'.join([row[0], row[1], ' '.join(pred), str(int(ok))]) + '\n')
        print(f'{split:>5}: word acc {result["acc"]:.3f} ({len(ds)} items)')

    # each eval item's nearest training lemmas in the model's hidden space
    train = splits['train']
    for split in ('ftune', 'dev', 'test'):
        ds = splits[split]
        sims, idxs = model.nearest(ds, train, k=args.k, pool=args.pool)
        with open(f'{run}/{split}.neighbors.tsv', 'w', encoding='utf-8') as f:
            f.write('src\ttgt\trank\tneighbor_src\tneighbor_tgt\tcosine\n')
            for row, row_sims, row_idxs in zip(ds.rows, sims.tolist(), idxs.tolist()):
                for rank, (sim, idx) in enumerate(zip(row_sims, row_idxs), 1):
                    near = train.rows[idx]
                    f.write('\t'.join([row[0], row[1], str(rank), near[0], near[1],
                                       f'{sim:.4f}']) + '\n')

    # does the k=1 neighbour's lemma -> past change, applied to the eval
    # lemma, reproduce what the model produced? under each representation of
    # the model's, and against surface and random neighbours as baselines
    analogies = {}
    for split in ('ftune', 'dev', 'test'):
        ds = splits[split]
        analogies[split] = {}
        with open(f'{run}/{split}.analogy.tsv', 'w', encoding='utf-8') as f:
            f.write('neighbors\tsrc\tgold\tpred\tneighbor_src\tneighbor_tgt'
                    '\tanalogy\tregular\tmatches_pred\tmatches_gold\n')
            for name, idxs in analogy.neighbor_sets(model, ds, train,
                                                    seed=args.seed).items():
                rows = analogy.analogize(ds, train, idxs, preds[split])
                analogies[split][name] = analogy.summarize(rows)
                for r in rows:
                    f.write('\t'.join([name, ' '.join(r['src']), ' '.join(r['gold']),
                                       ' '.join(r['pred']), ' '.join(r['neighbor_src']),
                                       ' '.join(r['neighbor_tgt']),
                                       ' '.join(r['analogy']) if r['applies'] else 'NA',
                                       str(int(r['regular'])), str(int(r['matches_pred'])),
                                       str(int(r['matches_gold']))]) + '\n')

    print('\ntest: k=1 analogy matches the model\'s prediction '
          '(regular / irregular gold)')
    for name, summ in analogies['test'].items():
        reg, irr = summ['regular'], summ['irregular']
        print(f'{name:>11}: {summ["all"]["matches_pred"]:.3f}  '
              f'({reg["matches_pred"]:.3f} of {reg["n"]} / '
              f'{irr["matches_pred"]:.3f} of {irr["n"]})  '
              f'rule applies {summ["all"]["applies"]:.3f}')

    # written last, so its presence marks a finished run
    with open(f'{run}/scores.json', 'w') as f:
        json.dump({'args': vars(args), **(info or {}), 'acc': scores,
                   'analogy': analogies}, f, indent=2)

def main(argv=None):
    args = build_parser().parse_args(argv)
    model, splits, info = train(args)
    print(f'best ftune loss {info["best_ftune_loss"]:.4f} at epoch '
          f'{info["best_epoch"]} of {info["epochs_run"] + 1}')
    analyze(model, splits, args, info)

if __name__ == '__main__':
    main()
