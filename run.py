import os
import argparse
import numpy as np
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


def aggregated(mod, data):
    mod.train(verbose=0)
    mod.get_metrics(data)
    mod.apply_op(np.mean, display_scores=True)
    mod.apply_op(np.std, display_scores=True)


def unsup_ae(args):
    data = get_data(args)
    mod = Aggregator(UnsupNN, args.number, data, args.size, reconstruct_loss=args.loss)
    aggregated(mod, data)


def sup_cats_ae(args):
    data = get_data(args)
    mod = Aggregator(SupNN, args.number, data, args.size)
    aggregated(mod, data)


def bin_ae(args):
    data = get_data(args)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=False)
    aggregated(mod, data)


def cats_ae(args):
    data = get_data(args)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=True)
    aggregated(mod, data)


def tsne(args):
    data = get_data(args)
    mod = Aggregator(TSNE, args.number, data, args.size)
    aggregated(mod, data)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(dest='method')
subparsers.required = True
parser.add_argument('datapath', type=str)
parser.add_argument('--size', type=int, default=2, help='Reduce data to this dimension')
parser.add_argument('--number', type=int, default=5, help='Number of models used to average')


# pca
parser_pca = subparsers.add_parser('pca', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_pca.set_defaults(func=pca)


# SemiSupNN arguments_parent
parent_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parent_parser.add_argument("--reconstruct_loss", type=str, help='One of "binary-crossentropy", "mse".',
                           default='binary-crossentropy')
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
parser_tsne.set_defaults(func=tsne)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
