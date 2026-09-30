import argparse
import copy
import json
import os

# cuBLAS refuses deterministic mode on GPU unless this is set before CUDA
# initialises; harmless on CPU
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
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

STOPS = ('loss', 'loss_irregular', 'loss_train_irregular')

def build_parser():
    parser = argparse.ArgumentParser(
        description='train a SegStrTransducer on one seed and size of the '
                    'English past tense data, stopping early on ftune loss and '
                    'train irregular accuracy (see --stop), then score every '
                    'split and run the analogy analysis')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--data', default='../data')
    parser.add_argument('--out', default='../results')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper bound; early stopping usually ends sooner')
    parser.add_argument('--patience', type=int, default=20,
                        help='epochs without improvement before stopping')
    parser.add_argument('--stop', choices=STOPS, default='loss_train_irregular',
                        help='loss: stop when ftune loss stops falling and keep '
                             'the lowest-loss epoch. loss_irregular: keep going '
                             'while either ftune loss falls or ftune irregular '
                             'accuracy rises, and keep the epoch with the best '
                             'ftune irregular accuracy, ties to the lower loss. '
                             'loss_train_irregular: keep going while either '
                             'ftune loss falls or train irregular accuracy '
                             'rises, and keep the lowest-loss epoch among those '
                             'with train irregular accuracy >= '
                             '--train_irr_target (or, if none reach it, the '
                             'epoch with the highest)')
    parser.add_argument('--train_irr_target', type=float, default=0.9,
                        help='train irregular accuracy an epoch needs before '
                             'loss_train_irregular will prefer it on loss')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--source_weight', type=float, default=0.0)
    parser.add_argument('--pool', choices=analogy.MODEL_POOLS, default='sep',
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

def irregular(ds):
    """
    the rows of `ds` whose gold past tense is irregular, as a dataset on the
    same vocab
    """
    rows = [row for row, src, tgt in zip(ds.rows, ds.srcs, ds.tgts)
            if analogy.classify(src, tgt) not in analogy.REGULAR]
    return SegStrPairDataset(ds.path, vocab=ds.vocab, rows=rows)

class IrregularAccuracy(Callback):
    """
    logs word accuracy on the irregular rows of each named dataset after every
    validation epoch, as <name>_irr_acc. irregulars are under a tenth of the
    data, so ftune loss says little about them; this is what lets stopping
    watch them directly. train_irr_acc is logged for inspection only
    """
    def __init__(self, datasets):
        self.datasets = {name: ds for name, ds in datasets.items() if len(ds)}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for name, ds in self.datasets.items():
            pl_module.log(f'{name}_irr_acc', pl_module.evaluate(ds)['acc'],
                          on_epoch=True)

class StopAndSelect(Callback):
    """
    early stopping and checkpoint selection in one, so both use the same
    notion of better. `mode` is one of STOPS:

    - loss: ordinary early stopping on ftune loss, keeping the lowest-loss
      epoch.
    - loss_irregular: patience resets on a new low in ftune loss OR a new high
      in ftune irregular accuracy; keeps the epoch with the highest ftune
      irregular accuracy, ties to the lower loss. ftune's irregulars are
      held-out verbs the model rarely gets, so this mostly chases noise.
    - loss_train_irregular: patience resets on a new low in ftune loss OR a
      new high in train irregular accuracy, so training runs on while the
      model is still memorising its irregulars, which ftune loss barely
      registers. keeps the lowest-loss epoch among those whose train
      irregular accuracy reaches `target`, or, if none does, the epoch with
      the highest train irregular accuracy (ties to the lower loss). the key
      (min(train_irr, target), -loss) does both in one comparison.

    reads the metrics in on_validation_end, once Lightning has reduced them
    over the epoch, as the built-in EarlyStopping does
    """
    def __init__(self, patience, mode='loss', target=0.9):
        if mode not in STOPS:
            raise ValueError(f'mode must be one of {STOPS}, not {mode!r}')
        self.patience = patience
        self.mode = mode
        self.target = target
        self.best_loss = float('inf')
        self.best_irr = -1.0
        self.wait = 0
        self.best_key = None
        self.best_state = None
        self.best = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        loss = metrics['val_loss'].item()
        ftune_irr = (metrics['ftune_irr_acc'].item()
                     if 'ftune_irr_acc' in metrics else None)
        train_irr = (metrics['train_irr_acc'].item()
                     if 'train_irr_acc' in metrics else None)
        # the irregular accuracy this mode watches, if it watches one
        irr = {'loss': None, 'loss_irregular': ftune_irr,
               'loss_train_irregular': train_irr}[self.mode]

        improved = loss < self.best_loss
        self.best_loss = min(self.best_loss, loss)
        if irr is not None:
            improved = improved or irr > self.best_irr
            self.best_irr = max(self.best_irr, irr)
        self.wait = 0 if improved else self.wait + 1

        if irr is None:
            key = (-loss,)
        elif self.mode == 'loss_train_irregular':
            key = (min(irr, self.target), -loss)
        else:
            key = (irr, -loss)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in pl_module.state_dict().items()})
            self.best = {'best_epoch': trainer.current_epoch,
                         'best_ftune_loss': loss, 'best_ftune_irr_acc': ftune_irr,
                         'best_train_irr_acc': train_irr}
        if self.wait >= self.patience:
            trainer.should_stop = True

def train(args):
    """
    trains until the `--stop` criterion has not improved for `patience`
    epochs, then restores the weights of the epoch it selected and saves them
    with everything needed to rebuild the model (see `load_run`)
    """
    L.seed_everything(args.seed)
    splits, vocab, max_len = load(paths(args.data, args.seed, args.size))
    model = build(args, splits, vocab, max_len)

    run = run_dir(args.out, args.seed, args.size)
    os.makedirs(run, exist_ok=True)
    track = IrregularAccuracy({'ftune': irregular(splits['ftune']),
                               'train': irregular(splits['train'])})
    stop = StopAndSelect(args.patience, mode=args.stop, target=args.train_irr_target)
    # track must run first, so its metrics exist by the time stop reads them
    trainer = L.Trainer(max_epochs=args.epochs, logger=CSVLogger(run, name=''),
                        callbacks=[track, stop], enable_checkpointing=False,
                        deterministic=True, enable_progress_bar=not args.quiet,
                        enable_model_summary=not args.quiet,
                        log_every_n_steps=1)
    trainer.fit(model)
    model.load_state_dict(stop.best_state)

    info = {**stop.best, 'epochs_run': trainer.current_epoch, 'stop': args.stop}
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

def analyze_analogy(model, splits, preds, run, methods=analogy.METHODS,
                    seed=0, suffix='') -> dict:
    """
    does the k=1 neighbour's lemma -> past change, applied to an eval lemma,
    reproduce what the model produced? under each of `methods` for ftune,
    dev and test, with training rows as the neighbours. `preds` holds the
    model's outputs for every split, train included. writes
    {split}.analogy{suffix}.tsv and returns the summaries
    """
    summaries = {}
    for split in ('ftune', 'dev', 'test'):
        rows = analogy.run_methods(model, splits[split], splits['train'],
                                   preds[split], preds['train'],
                                   methods=methods, seed=seed)
        analogy.write_tsv(f'{run}/{split}.analogy{suffix}.tsv', rows)
        summaries[split] = {name: analogy.summarize(r) for name, r in rows.items()}
    return summaries

def print_analogy(summaries, label='test') -> None:
    """
    one line per method: exact-match and class agreement with the model's
    prediction, overall and on items where the prediction is irregular
    """
    print(f'\n{label}: k=1 analogy vs the model\'s prediction')
    print(f'{"":>21}  {"exact":>6}  {"class":>6}  {"kappa":>6}  {"irreg class":>12}')
    for name, summ in summaries.items():
        every, irr = summ['all'], summ['pred_irregular']
        print(f'{name:>21}  {every["matches_pred"]:6.3f}  {every["class_acc_pred"]:6.3f}  '
              f'{every["kappa_pred"]:6.3f}  {irr["class_acc_pred"]:6.3f} of {irr["n"]:<3}')

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
        sims, idxs = model.nearest(ds, train, k=args.k, pool=args.pool,
                                   query_preds=preds[split], ref_preds=preds['train'])
        with open(f'{run}/{split}.neighbors.tsv', 'w', encoding='utf-8') as f:
            f.write('src\ttgt\trank\tneighbor_src\tneighbor_tgt\tcosine\n')
            for row, row_sims, row_idxs in zip(ds.rows, sims.tolist(), idxs.tolist()):
                for rank, (sim, idx) in enumerate(zip(row_sims, row_idxs), 1):
                    near = train.rows[idx]
                    f.write('\t'.join([row[0], row[1], str(rank), near[0], near[1],
                                       f'{sim:.4f}']) + '\n')

    analogies = analyze_analogy(model, splits, preds, run, seed=args.seed)
    print_analogy(analogies['test'])

    # written last, so its presence marks a finished run
    with open(f'{run}/scores.json', 'w') as f:
        json.dump({'args': vars(args), **(info or {}), 'acc': scores,
                   'analogy': analogies}, f, indent=2)

def main(argv=None):
    args = build_parser().parse_args(argv)
    model, splits, info = train(args)
    fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'
    print(f'selected epoch {info["best_epoch"]} of {info["epochs_run"]}: ftune loss '
          f'{info["best_ftune_loss"]:.4f}, ftune irregular acc '
          f'{fmt(info["best_ftune_irr_acc"])}, train irregular acc '
          f'{fmt(info["best_train_irr_acc"])}')
    analyze(model, splits, args, info)

if __name__ == '__main__':
    main()
