#!/usr/bin/env python
"""
Tox21 data training
Additionally, artificial pyridine dataset training is supported.
"""

from __future__ import print_function

import logging
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import numpy as np
import argparse
import chainer
from chainer import functions as F
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E
import json
from rdkit import RDLogger, Chem

from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry import datasets as D
from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator  # NOQA
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import predictor

import data


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'nfpdrop', 'ggnndrop']
    label_names = D.get_tox21_label_names() + ['pyridine']
    iterator_type = ['serial', 'balanced']

    parser = argparse.ArgumentParser(
        description='Multitask Learning with Tox21.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp', help='graph convolution model to use '
                        'as a predictor.')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='', help='target label for logistic '
                        'regression. Use all labels if this option '
                        'is not specified.')
    parser.add_argument('--iterator-type', type=str, choices=iterator_type,
                        default='serial', help='iterator type. If `balanced` '
                        'is specified, data is sampled to take same number of'
                        'positive/negative labels during training.')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='number of mlp layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                        'not to use GPU and to run the code in CPU.')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to output directory')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='path to a trainer snapshot')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--dropout-ratio', '-d', type=float, default=0.25,
                        help='dropout_ratio')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--num-train', type=int, default=-1,
                        help='number of training data to be used, '
                             'negative value indicates use all train data')
    args = parser.parse_args()

    method = args.method
    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        class_num = len(label_names)

    # Dataset preparation
    train, val, test, train_smiles, val_smiles, test_smiles = data.load_dataset(method, labels)

    num_train = args.num_train  # 100
    if num_train > 0:
        # reduce size of train data
        seed = args.seed  # 0
        np.random.seed(seed)
        train_selected_label = np.random.permutation(np.arange(len(train)))[:num_train]
        print('num_train', num_train, len(train_selected_label), train_selected_label)
        train = NumpyTupleDataset(*train.features[train_selected_label, :])
    # Network
    predictor_ = predictor.build_predictor(
        method, args.unit_num, args.conv_layers, class_num, args.dropout_ratio,
        args.n_layers
    )

    iterator_type = args.iterator_type
    if iterator_type == 'serial':
        train_iter = I.SerialIterator(train, args.batchsize)
    elif iterator_type == 'balanced':
        if class_num > 1:
            raise ValueError('BalancedSerialIterator can be used with only one'
                             'label classification, please specify label to'
                             'be predicted by --label option.')
        train_iter = BalancedSerialIterator(
            train, args.batchsize, train.features[:, -1], ignore_labels=-1)
        train_iter.show_label_stats()
    else:
        raise ValueError('Invalid iterator type {}'.format(iterator_type))
    val_iter = I.SerialIterator(val, args.batchsize,
                                repeat=False, shuffle=False)
    classifier = L.Classifier(predictor_,
                              lossfun=F.sigmoid_cross_entropy,
                              accfun=F.binary_accuracy)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(classifier)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(val_iter, classifier,
                               device=args.gpu, converter=concat_mols))
    trainer.extend(E.LogReport())

    # --- ROCAUC Evaluator ---
    train_eval_iter = I.SerialIterator(train, args.batchsize,
                                       repeat=False, shuffle=False)
    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor_,
        device=args.gpu, converter=concat_mols, name='train'))
    trainer.extend(ROCAUCEvaluator(
        val_iter, classifier, eval_func=predictor_,
        device=args.gpu, converter=concat_mols, name='val'))
    trainer.extend(E.PrintReport([
        'epoch', 'main/loss', 'main/accuracy', 'train/main/roc_auc',
        'validation/main/loss', 'validation/main/accuracy',
        'val/main/roc_auc', 'elapsed_time']))

    trainer.extend(E.ProgressBar(update_interval=10))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    chainer.serializers.save_npz(
        os.path.join(args.out, 'predictor.npz'), predictor_)


if __name__ == '__main__':
    # Disable errors by RDKit occurred in preprocessing Tox21 dataset.
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # show INFO level log from chainer chemistry
    logging.basicConfig(level=logging.INFO)

    main()
