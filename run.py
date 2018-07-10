import os
import argparse
import json
from data import Data
from models import *
from classifiers import test_model, Aggregator


def get_data(args):
    return Data(args.datapath + os.sep + 'UNSW-NB15_all.csv',
                args.datapath + os.sep + 'UNSW_NB15_training-set.csv',
                args.datapath + os.sep + 'UNSW_NB15_testing-set.csv')


def pca(args):
    data = get_data(args)
    mod = PCA(data, args.size)
    mod.train()
    test_model(data, mod)


def _aggregated(mod, data, verbose):
    if verbose:
        mod.train(verbose=2, **args.train)  # force verbose=2 for keras models
    else:
        mod.train(**args.train)
    mod.get_metrics(data)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def aggregated(mod, data):
    _aggregated(mod, data, False)


def aggregated_keras(mod, data):
    _aggregated(mod, data, True)


def unsup_ae(args):
    data = get_data(args)
    mod = Aggregator(UnsupNN, args.number, data, args.size,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay)
    aggregated_keras(mod, data)


def sup_cats_ae(args):
    data = get_data(args)
    mod = Aggregator(SupNN, args.number, data, args.size,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay)
    aggregated_keras(mod, data)
    mod.get_own_metrics(data, categories=True)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def bin_ae(args):
    data = get_data(args)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=False,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay)
    aggregated_keras(mod, data)
    mod.get_own_metrics(data, categories=False)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def cats_ae(args):
    data = get_data(args)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=True,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay)
    aggregated_keras(mod, data)
    mod.get_own_metrics(data, categories=True)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def tsne(args):
    data = get_data(args)
    mod = Aggregator(TSNE, args.number, data, args.size,
                     perplexity=args.perplexity, early_exaggeration=args.early_exaggeration,
                     learning_rate=args.learning_rate, n_iter=args.n_iter,
                     n_iter_without_progress=args.n_iter_without_progress,
                     min_grad_norm=args.min_grad_norm, metric=args.metric,
                     init=args.init, verbose=args.verbose, random_state=args.random_state,
                     method=args.method, angle=args.angle
                     )
    aggregated(mod, data)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(dest='method')
subparsers.required = True
parser.add_argument('datapath', type=str)
parser.add_argument('--size', type=int, default=2, help='Reduce data to this dimension')
parser.add_argument('--number', type=int, default=5, help='Number of models used to average')
parser.add_argument('--train', type=json.loads, default={},
                    help='Dictionary (in the form of string) that is passed to the train command as arguments')


# pca
parser_pca = subparsers.add_parser('pca', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_pca.set_defaults(func=pca)


# SemiSupNN arguments_parent
parent_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parent_parser.add_argument("--reconstruct_loss", type=str, help='One of "binary_crossentropy", "mse".',
                           default='binary_crossentropy')
parent_parser.add_argument('--reconstruct_weight', type=float, default=0.1,
                           help='Weight of reconstruction loss. Label loss is always 1.0')
parent_parser.add_argument('--enc_regularizer_weight', type=float, default=0.1,
                           help='Weight of l2 regularization in the encoder')
parent_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parent_parser.add_argument('--lr_decay', type=float, default=1e-5, help='Decay of learning rate')
parent_parser.add_argument('--dec_regularizer_weight', type=float, default=0.,
                           help='Weight of the l2 regularization in the decoder')


# unsup_ae
parser_unsup_ae = subparsers.add_parser('unsup_ae', parents=[parent_parser],
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_unsup_ae.set_defaults(func=unsup_ae)


# sup_cats_ae
parser_sup_cats_ae = subparsers.add_parser('sup_cats_ae', parents=[parent_parser],
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_sup_cats_ae.set_defaults(func=sup_cats_ae)


# bin_ae
parser_bin_ae = subparsers.add_parser('bin_ae', parents=[parent_parser],
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_bin_ae.set_defaults(func=bin_ae)


# cats_ae
parser_cats_ae = subparsers.add_parser('cats_ae', parents=[parent_parser],
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_cats_ae.set_defaults(func=cats_ae)


# tsne
parser_tsne = subparsers.add_parser('tsne', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_tsne.add_argument('--perplexity', type=float, default=30.0)
parser_tsne.add_argument('--early_exaggeration', type=float, default=30.0)
parser_tsne.add_argument('--learning_rate', type=float, default=200.0)
parser_tsne.add_argument('--n_iter', type=int, default=1000)
parser_tsne.add_argument('--n_iter_without_progress', type=int, default=300)
parser_tsne.add_argument('--min_grad_norm', type=float, default=1e-07)
parser_tsne.add_argument('--metric', type=str, default='euclidean')
parser_tsne.add_argument('--init', type=str, default='random')
parser_tsne.add_argument('--verbose', type=int, default=0)
parser_tsne.add_argument('--random_state', type=int, default=None)
parser_tsne.add_argument('--method', type=str, default='barnes_hut')
parser_tsne.add_argument('--angle', type=float, default=0.5)
parser_tsne.set_defaults(func=tsne)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
