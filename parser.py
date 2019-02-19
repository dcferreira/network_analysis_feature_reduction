import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np

from classifiers import Aggregator
from models import PCA, LDA, UnsupNN, SupNN, SemisupNN, DeepSemiSupNN, TSNE, MDS, DeepUnsupNN


def set_seeds(tf_seed=False, pytorch_seed=False):
    random.seed(1337)
    np.random.seed(1337)
    if tf_seed:
        import tensorflow
        tensorflow.set_random_seed(1337)
    if pytorch_seed:
        import torch
        torch.manual_seed(1337)


def pca(args, data):
    set_seeds()
    mod = Aggregator(PCA, 1, data, args.size)
    mod.train()
    mod.get_metrics(data)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def lda(args, data):
    set_seeds()
    mod = Aggregator(LDA, 1, data, args.size, categories=args.categories)
    mod.train()
    mod.get_metrics(data)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def _aggregated(args, mod, data, verbose, path=None):
    if path is not None and os.path.exists(path):  # check if models have already been trained
        print('Loading existing models...')
        mod.load_models(path)
    else:
        if verbose:
            mod.train(verbose=2, **args.train)  # force verbose=2 for keras models
        else:
            mod.train(**args.train)
    mod.get_metrics(data)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)
    if path is not None:
        mod.save_models(path)


def aggregated(args, mod, data):
    _aggregated(args, mod, data, False)


def aggregated_keras(args, mod, data, path):
    _aggregated(args, mod, data, True, path)


def unsup_ae(args, data):
    set_seeds(tf_seed=True)
    mod = Aggregator(UnsupNN, args.number, data, args.size,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay, encoder_regularizer=args.encoder_regularizer)
    aggregated_keras(args, mod, data, args.model_path)


def deep_unsup_ae(args, data):
    set_seeds(pytorch_seed=True)
    mod = Aggregator(DeepUnsupNN, args.number, data, args.size, categories=False,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     lr=args.lr, lr_decay=args.lr_decay, random_seed=args.random_seed,
                     checkpoint_path=args.checkpoint)
    aggregated_keras(args, mod, data, args.model_path)


def sup_cats_ae(args, data):
    set_seeds(tf_seed=True)
    mod = Aggregator(SupNN, args.number, data, args.size,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay, encoder_regularizer=args.encoder_regularizer)
    aggregated_keras(args, mod, data, args.model_path)
    mod.get_own_metrics(data, categories=True)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def bin_ae(args, data):
    set_seeds(tf_seed=True)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=False,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay, encoder_regularizer=args.encoder_regularizer)
    aggregated_keras(args, mod, data, args.model_path)
    mod.get_own_metrics(data, categories=False)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def deep_bin_ae(args, data):
    set_seeds(pytorch_seed=True)
    mod = Aggregator(DeepSemiSupNN, args.number, data, args.size, categories=False,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     lr=args.lr, lr_decay=args.lr_decay, random_seed=args.random_seed,
                     checkpoint_path=args.checkpoint)
    aggregated_keras(args, mod, data, args.model_path)
    mod.get_own_metrics(data, categories=False)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def cats_ae(args, data):
    set_seeds(tf_seed=True)
    mod = Aggregator(SemisupNN, args.number, data, args.size, categories=True,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     enc_regularizer_weight=args.enc_regularizer_weight,
                     dec_regularizer_weight=args.dec_regularizer_weight,
                     lr=args.lr, lr_decay=args.lr_decay, encoder_regularizer=args.encoder_regularizer)
    aggregated_keras(args, mod, data, args.model_path)
    mod.get_own_metrics(data, categories=True)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def deep_cats_ae(args, data):
    set_seeds(pytorch_seed=True)
    mod = Aggregator(DeepSemiSupNN, args.number, data, args.size, categories=True,
                     reconstruct_loss=args.reconstruct_loss, reconstruct_weight=args.reconstruct_weight,
                     lr=args.lr, lr_decay=args.lr_decay, random_seed=args.random_seed,
                     checkpoint_path=args.checkpoint)
    aggregated_keras(args, mod, data, args.model_path)
    mod.get_own_metrics(data, categories=True)
    mod.mean(display_scores=True)
    mod.std(display_scores=True)


def tsne(args, data):
    set_seeds()
    mod = Aggregator(TSNE, args.number, data, args.size,
                     perplexity=args.perplexity, early_exaggeration=args.early_exaggeration,
                     learning_rate=args.learning_rate, n_iter=args.n_iter,
                     n_iter_without_progress=args.n_iter_without_progress,
                     min_grad_norm=args.min_grad_norm, metric=args.metric,
                     init=args.init, verbose=args.verbose, random_state=args.random_state,
                     method=args.method, angle=args.angle
                     )
    _aggregated(args, mod, data, False, args.model_path)


def mds(args, data):
    set_seeds()
    mod = Aggregator(MDS, args.number, data, args.size, n_components=args.size,
                     metric=args.metric, n_init=args.n_init, max_iter=args.max_iter,
                     verbose=args.verbose, eps=args.eps, n_jobs=args.n_jobs,
                     random_state=args.random_state, dissimilarity=args.dissimilarity
                     )
    aggregated(args, mod, data)


def get_weights(args, data):
    weights1 = OrderedDict([('model', []), ('param', [])])
    weights2 = OrderedDict([('model', []), ('param', [])])
    for feat in data.columns:
        weights1[feat] = []
        weights2[feat] = []

    # bin
    if args.bin_ae_path is not None:
        for w in args.bin_ae_weight:
            model_path = args.bin_ae_path % w
            assert os.path.exists(model_path), 'model %s doesn\'t exist!' % model_path
            agg = Aggregator(SemisupNN, 5, data, 2, categories=False, reconstruct_loss='mse',
                             reconstruct_weight=float(w))
            agg.load_models(model_path)
            for mod in agg.models:
                weights1['model'].append('bin_ae')
                weights1['param'].append(w)
                weights2['model'].append('bin_ae')
                weights2['param'].append(w)
                for name, (w1, w2) in zip(data.columns, mod.get_feature_weights()):
                    weights1[name].append(w1)
                    weights2[name].append(w2)

    # cats
    if args.cats_ae_path is not None:
        for w in args.cats_ae_weight:
            model_path = args.cats_ae_path % w
            assert os.path.exists(model_path), 'model %s doesn\'t exist!' % model_path
            agg = Aggregator(SemisupNN, 5, data, 2, categories=True, reconstruct_loss='mse',
                             reconstruct_weight=float(w))
            agg.load_models(model_path)
            for mod in agg.models:
                weights1['model'].append('cats_ae')
                weights1['param'].append(w)
                weights2['model'].append('cats_ae')
                weights2['param'].append(w)
                for name, (w1, w2) in zip(data.columns, mod.get_feature_weights()):
                    weights1[name].append(w1)
                    weights2[name].append(w2)

    print('Weights 1')
    for k, v in weights1.items():
        print(k, ','.join([str(vv) for vv in v]), sep=',')

    print('Weights 2')
    for k, v in weights2.items():
        print(k, ','.join([str(vv) for vv in v]), sep=',')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(dest='method', help='Feature reduction method to use.')
subparsers.required = True
parser.add_argument('--size', type=int, default=2, help='Reduce data to this dimension')
parser.add_argument('--number', type=int, default=5, help='Number of models used to average')
parser.add_argument('--train', type=json.loads, default={},
                    help='Dictionary (in the form of string) that is passed to the train command as arguments')


# pca
parser_pca = subparsers.add_parser('pca', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_pca.set_defaults(func=pca)


# lda
parser_lda = subparsers.add_parser('lda', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_lda.add_argument('--categories', action='store_true',
                        help='Whether to use categories or only binary labels for the LDA.')
parser_lda.set_defaults(func=lda)


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
parent_parser.add_argument('--encoder_regularizer', type=str, default='l2',
                           help='Type of regularizer used for the encoder (l2, l1)')
parent_parser.add_argument('--model_path', type=str, default=None,
                           help='Path in which to save the model.')

# DeepSemiSupNN arguments
deepsemisup_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
deepsemisup_parser.add_argument("--reconstruct_loss", type=str, help='One of "binary_crossentropy", "mse".',
                                default='binary_crossentropy')
deepsemisup_parser.add_argument('--reconstruct_weight', type=float, default=0.1,
                                help='Weight of reconstruction loss. Label loss is always 1.0')
deepsemisup_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
deepsemisup_parser.add_argument('--lr_decay', type=float, default=1e-5, help='Decay of learning rate')
deepsemisup_parser.add_argument('--random_seed', type=int, default=1337, help='Random seed used for training')
deepsemisup_parser.add_argument('--checkpoint', type=str, default='tmp/',
                                help='Path in which to save the model\'s checkpoints while training.')
deepsemisup_parser.add_argument('--model_path', type=str, default=None,
                                help='Path in which to save the model.')

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

# deep_bin_ae
parser_deep_bin_ae = subparsers.add_parser('deep_bin_ae', parents=[deepsemisup_parser],
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_deep_bin_ae.set_defaults(func=deep_bin_ae)

# cats_ae
parser_cats_ae = subparsers.add_parser('cats_ae', parents=[parent_parser],
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_cats_ae.set_defaults(func=cats_ae)

# deep_cats_ae
parser_deep_cats_ae = subparsers.add_parser('deep_cats_ae', parents=[deepsemisup_parser],
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_deep_cats_ae.set_defaults(func=deep_cats_ae)

# deep_unsup_ae
parser_deep_unsup_ae = subparsers.add_parser('deep_unsup_ae', parents=[deepsemisup_parser],
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_deep_unsup_ae.set_defaults(func=deep_unsup_ae)


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
parser_tsne.add_argument('--model_path', type=str, default=None,
                         help='Path in which to save the model.')
parser_tsne.set_defaults(func=tsne)


# multidimensional scaling (mds)
parser_mds = subparsers.add_parser('mds', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_mds.add_argument('--metric', type=bool, default=True)
parser_mds.add_argument('--n_init', type=int, default=4)
parser_mds.add_argument('--max_iter', type=int, default=300)
parser_mds.add_argument('--verbose', type=int, default=0)
parser_mds.add_argument('--eps', type=float, default=1e-3)
parser_mds.add_argument('--n_jobs', type=int, default=1)
parser_mds.add_argument('--random_state', type=int, default=None)
parser_mds.add_argument('--dissimilarity', type=str, default='euclidean')
parser_mds.set_defaults(func=mds)


# getting weights
parser_weights = subparsers.add_parser('weights', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_weights.add_argument('--bin_ae_path', type=str, default=None)
parser_weights.add_argument('--bin_ae_weight', type=str, nargs='+')
parser_weights.add_argument('--cats_ae_path', type=str, default=None)
parser_weights.add_argument('--cats_ae_weight', type=str, nargs='+')
parser_weights.set_defaults(func=get_weights)
