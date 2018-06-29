"""
Calculate statistics by seeds.
"""
import argparse

import matplotlib
import pandas

matplotlib.use('agg')
import matplotlib.pyplot as plt

from chainer import functions as F
from chainer import links as L
from chainer import serializers
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
import numpy as np
from rdkit import RDLogger, Chem
from sklearn.metrics import auc
from tqdm import tqdm

import sys
import os
import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import predictor
from saliency.calculator.gradient_calculator import GradientCalculator

import data
from data import PYRIDINE_SMILES
from plot_precision_recall import calc_recall_precision


def calc_prc_auc(saliency, rates, haspiindex):
    recall, precision = calc_recall_precision(saliency, rates, haspiindex)
    print('recall', recall)
    print('precision', precision)
    prcauc = auc(recall, precision)
    print('prcauc', prcauc)
    return recall, precision, prcauc


def parse():
    parser = argparse.ArgumentParser(
        description='Multitask Learning with Tox21.')
    parser.add_argument('--method', '-m', type=str,
                        default='nfp', help='graph convolution model to use '
                                            'as a predictor.')
    parser.add_argument('--label', '-l', type=str,
                        default='pyridine', help='target label for logistic '
                        'regression. Use all labels if this option '
                        'is not specified.')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='number of mlp layers')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
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
    parser.add_argument('--dropout-ratio', '-d', type=float, default=0.25,
                        help='dropout_ratio')
    parser.add_argument('--seeds', type=int, default=0,
                        help='number of seed to use for calculation')
    parser.add_argument('--num-train', type=int, default=-1,
                        help='number of training data to be used, '
                             'negative value indicates use all train data')
    parser.add_argument('--scale', type=float, default=0.15, help='scale for smoothgrad')
    parser.add_argument('--mode', type=str, default='absolute', help='mode for smoothgrad')
    args = parser.parse_args()
    return args


def main(method, labels, unit_num, conv_layers, class_num, n_layers,
         dropout_ratio, model_path_list, save_dir_path, scale=0.15, mode='relative', M=5):
    # Dataset preparation
    train, val, test, train_smiles, val_smiles, test_smiles = data.load_dataset(method, labels)

    # --- model preparation ---
    model = predictor.build_predictor(
        method, unit_num, conv_layers, class_num, dropout_ratio, n_layers)

    classifier = L.Classifier(model,
                              lossfun=F.sigmoid_cross_entropy,
                              accfun=F.binary_accuracy)

    target_dataset = val
    target_smiles = val_smiles

    val_mols = [Chem.MolFromSmiles(smi) for smi in tqdm(val_smiles)]

    pi = Chem.MolFromSmarts(PYRIDINE_SMILES)
    piindex = np.where(np.array([mol.HasSubstructMatch(pi) for mol in val_mols]) == True)
    haspi = np.array(val_mols)[piindex]

    # It only extracts one substructure, not expected behavior
    # haspiindex = [set(mol.GetSubstructMatch(pi)) for mol in haspi]
    def flatten_tuple(x):
        return [element for tupl in x for element in tupl]
    haspiindex = [flatten_tuple(mol.GetSubstructMatches(pi)) for mol in haspi]
    print('piindex', piindex)
    print('haspi', haspi.shape)
    print('haspiindex', haspiindex)
    print('haspiindex length', [len(k) for k in haspiindex])

    pyrigine_dataset = NumpyTupleDataset(*target_dataset.features[piindex, :])
    pyrigine_smiles = target_smiles[piindex]

    atoms = pyrigine_dataset.features[:, 0]
    num_atoms = [len(a) for a in atoms]

    def clip_original_size(saliency, num_atoms):
        """`saliency` array is 0 padded, this method align to have original
        molecule's length
        """
        assert len(saliency) == len(num_atoms)
        saliency_list = []
        for i in range(len(saliency)):
            saliency_list.append(saliency[i, :num_atoms[i]])
        return saliency_list

    def preprocess_fun(*inputs):
        atom, adj, t = inputs
        # HACKING for now...
        # classifier.predictor.pick = True
        # result = classifier.predictor(atom, adj)
        atom_embed = classifier.predictor.graph_conv.embed(atom)
        return atom_embed, adj, t

    def eval_fun(*inputs):
        atom_embed, adj, t = inputs
        prob = classifier.predictor(atom_embed, adj)
        # print('embed', atom_embed.shape, 'prob', prob.shape)
        out = F.sum(prob)
        # return {'embed': atom_embed, 'out': out}
        return out

    gradient_calculator = GradientCalculator(
        classifier, eval_fun=eval_fun,
        # target_key='embed', eval_key='out',
        target_key=0,
    )

    print('M', M)
    # rates = np.array(list(range(1, 11))) * 0.1
    num = 20
    # rates = np.linspace(0, 1, num=num+1)[1:]
    rates = np.linspace(0.1, 1, num=num)
    print('rates', len(rates), rates)

    fig = plt.figure(figsize=(7, 5), dpi=200)

    precisions_vanilla = []
    precisions_smooth = []
    precisions_bayes = []
    precisions_bayes_smooth = []
    prcauc_vanilla = []
    prcauc_smooth = []
    prcauc_bayes = []
    prcauc_bayes_smooth = []

    prcauc_diff_smooth_vanilla = []
    prcauc_diff_bayes_vanilla = []
    for model_path in model_path_list:
        serializers.load_npz(model_path, model)

        # --- VanillaGrad ---
        saliency_arrays = gradient_calculator.compute_vanilla(
            pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun)
        saliency = gradient_calculator.transform(
            saliency_arrays, ch_axis=3, method='square')
        # saliency_arrays (1, 28, 43, 64) -> M, batch_size, max_atom, ch_dim
        print('saliency_arrays', saliency_arrays.shape)
        # saliency (28, 43) -> batch_size, max_atom
        print('saliency', saliency.shape)
        saliency_vanilla = clip_original_size(saliency, num_atoms)

        # recall & precision
        print('vanilla')
        naiverecall, naiveprecision, naiveprcauc = calc_prc_auc(saliency_vanilla, rates, haspiindex)
        precisions_vanilla.append(naiveprecision)
        prcauc_vanilla.append(naiveprcauc)

        # --- SmoothGrad ---
        saliency_arrays = gradient_calculator.compute_smooth(
            pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun,
            M=M, scale=scale, mode=mode)
        saliency = gradient_calculator.transform(
            saliency_arrays, ch_axis=3, method='square')

        saliency_smooth = clip_original_size(saliency, num_atoms)

        # recall & precision
        print('smooth')
        smoothrecall, smoothprecision, smoothprcauc = calc_prc_auc(saliency_smooth, rates, haspiindex)
        precisions_smooth.append(smoothprecision)
        prcauc_smooth.append(smoothprcauc)

        # --- BayesGrad ---
        saliency_arrays = gradient_calculator.compute_vanilla(
            pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun, train=True, M=M)
        saliency = gradient_calculator.transform(
            saliency_arrays, ch_axis=3, method='square', lam=0)
        saliency_bayes = clip_original_size(saliency, num_atoms)

        bgrecall0, bgprecision0, bayesprcauc = calc_prc_auc(saliency_bayes, rates, haspiindex)
        precisions_bayes.append(bgprecision0)
        prcauc_bayes.append(bayesprcauc)
        prcauc_diff_smooth_vanilla.append(smoothprcauc - naiveprcauc)
        prcauc_diff_bayes_vanilla.append(bayesprcauc - naiveprcauc)

        # --- BayesSmoothGrad ---
        saliency_arrays = gradient_calculator.compute_smooth(
            pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun,
            M=M, scale=scale, mode=mode, train=True)
        saliency = gradient_calculator.transform(
            saliency_arrays, ch_axis=3, method='square')
        saliency_bayes_smooth = clip_original_size(saliency, num_atoms)
        # recall & precision
        print('bayes smooth')
        bayes_smoothrecall, bayes_smoothprecision, bayes_smoothprcauc = calc_prc_auc(saliency_bayes_smooth, rates, haspiindex)
        precisions_bayes_smooth.append(bayes_smoothprecision)
        prcauc_bayes_smooth.append(bayes_smoothprcauc)

    precisions_vanilla = np.array(precisions_vanilla)
    precisions_smooth = np.array(precisions_smooth)
    precisions_bayes = np.array(precisions_bayes)
    precisions_bayes_smooth = np.array(precisions_bayes_smooth)

    df = pandas.DataFrame({
        'model_path': model_path,
        'prcauc_vanilla': prcauc_vanilla,
        'prcauc_smooth': prcauc_smooth,
        'prcauc_bayes': prcauc_bayes,
        'prcauc_bayes_smooth': prcauc_bayes_smooth,
        'prcauc_diff_smooth_vanilla': prcauc_diff_smooth_vanilla,
        'prcauc_diff_bayes_vanilla': prcauc_diff_bayes_vanilla
    })
    save_csv_path = save_dir_path + '/prcauc_{}_{}.csv'.format(mode, scale)
    print('save to ', save_csv_path)
    df.to_csv(save_csv_path)

    prcauc_vanilla = np.array(prcauc_vanilla)
    prcauc_smooth = np.array(prcauc_smooth)
    prcauc_bayes = np.array(prcauc_bayes)
    prcauc_bayes_smooth = np.array(prcauc_bayes_smooth)
    prcauc_diff_smooth_vanilla = np.array(prcauc_diff_smooth_vanilla)
    prcauc_diff_bayes_vanilla = np.array(prcauc_diff_bayes_vanilla)

    def show_avg_std(array, tag=''):
        print('{}: mean {:8.03}, std {:8.03}'
              .format(tag, np.mean(array, axis=0), np.std(array, axis=0)))
        return {'method': tag, 'mean': np.mean(array, axis=0), 'std': np.std(array, axis=0)}

    df = pandas.DataFrame([
        show_avg_std(prcauc_vanilla, tag='vanilla'),
        show_avg_std(prcauc_smooth, tag='smooth'),
        show_avg_std(prcauc_bayes, tag='bayes'),
        show_avg_std(prcauc_bayes_smooth, tag='bayes_smooth'),
        show_avg_std(prcauc_diff_smooth_vanilla, tag='diff_smooth_vanilla'),
        show_avg_std(prcauc_diff_bayes_vanilla, tag='diff_bayes_vanilla'),
    ])
    save_csv_path = save_dir_path + '/prcauc_stats_{}_{}.csv'.format(mode, scale)
    print('save to ', save_csv_path)
    df.to_csv(save_csv_path)
    # import IPython; IPython.embed()

    def _plot_with_errorbar(x, precisions, color='blue', alpha=None, label=None):
        y = np.mean(precisions, axis=0)
        plt.errorbar(x, y, yerr=np.std(precisions, axis=0), fmt='ro', ecolor=color)  # fmt=''
        plt.plot(x, y, 'k-', color=color, label=label, alpha=alpha)

    alpha = 0.5
    _plot_with_errorbar(rates, precisions_vanilla, color='blue', alpha=alpha, label='VanillaGrad')
    _plot_with_errorbar(rates, precisions_smooth, color='green', alpha=alpha, label='SmoothGrad')
    _plot_with_errorbar(rates, precisions_bayes, color='yellow', alpha=alpha, label='BayesGrad')
    _plot_with_errorbar(rates, precisions_bayes_smooth, color='orange', alpha=alpha, label='BayesSmoothGrad')
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    save_path = os.path.join(save_dir_path, 'artificial_pr.png')
    print('saved to ', save_path)
    plt.savefig(save_path)


if __name__ == '__main__':
    # Disable errors by RDKit occurred in preprocessing Tox21 dataset.
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # show INFO level log from chainer chemistry
    logging.basicConfig(level=logging.INFO)

    args = parse()
    # --- config ---
    method = args.method
    labels = args.label
    unit_num = args.unit_num
    conv_layers = args.conv_layers
    class_num = 1
    n_layers = args.n_layers
    dropout_ratio = args.dropout_ratio
    num_train = args.num_train
    seeds = args.seeds

    root = '.'
    model_path_list = []
    for i in range(seeds):
        dir_path = '{}/results/{}_{}_numtrain{}_seed{}'.format(
            root, method, labels, num_train, i)
        model_path = os.path.join(dir_path, 'predictor.npz')
        model_path_list.append(model_path)

    save_dir_path = '{}/results/{}_{}_numtrain{}_seed0-{}'.format(
        root, method, labels, num_train, seeds-1)
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # --- config end ---

    main(method, labels, unit_num, conv_layers, class_num, n_layers,
         dropout_ratio, model_path_list, save_dir_path, args.scale, args.mode, M=100)
