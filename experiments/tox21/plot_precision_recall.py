import sys
import os
import logging
import argparse
import json

try:
    import matplotlib
    matplotlib.use('agg')
except:
    pass
import matplotlib.pyplot as plt

from chainer import functions as F
from chainer import links as L
from tqdm import tqdm
from chainer import serializers
import numpy as np
from rdkit import RDLogger, Chem

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import predictor
from saliency.calculator.gradient_calculator import GradientCalculator
from saliency.calculator.integrated_gradients_calculator import IntegratedGradientsCalculator
from saliency.calculator.occlusion_calculator import OcclusionCalculator

import data
from data import PYRIDINE_SMILES, hassubst


def percentile_index(ar, num):
    """
    ar (numpy.ndarray): array
    num (float): rate

    Extract `num` rate of largest index in this array.
    """
    threshold = int(len(ar) * num)
    idx = np.argsort(ar)
    return idx[-threshold:]


def calc_recall_precision_for_rate(grads, rate, haspiindex):
    recall_list = []
    hit_rate_list = []
    for i in range(len(grads)):
        largest_index = percentile_index(grads[i], float(rate))
        set_largest_index = set(largest_index)
        hit_index = set_largest_index.intersection(haspiindex[i])
        hit_num = len(hit_index)
        hit_rate = float(hit_num) / float(len(set_largest_index))

        recall_list.append(float(hit_num) / len(haspiindex[i]))
        hit_rate_list.append(hit_rate)
    recall = np.mean(np.array(recall_list))
    precision = np.mean(np.array(hit_rate_list))
    return recall, precision


def calc_recall_precision(grads, rates, haspiindex):
    r_list = []
    p_list = []
    for rate in rates:
        r, p = calc_recall_precision_for_rate(grads, rate, haspiindex)
        r_list.append(r)
        p_list.append(p)
    return r_list, p_list


def parse():
    parser = argparse.ArgumentParser(
        description='Multitask Learning with Tox21.')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                             'not to use GPU and to run the code in CPU.')
    parser.add_argument('--dirpath', '-d', type=str, default='results',
                        help='path to train results directory')
    parser.add_argument('--calculator', type=str, default='gradient')
    args = parser.parse_args()
    return args


def main(method, labels, unit_num, conv_layers, class_num, n_layers,
         dropout_ratio, model_path, save_path):
    # Dataset preparation
    train, val, test, train_smiles, val_smiles, test_smiles = data.load_dataset(method, labels)

    # --- model preparation ---
    model = predictor.build_predictor(
        method, unit_num, conv_layers, class_num, dropout_ratio, n_layers)

    classifier = L.Classifier(model,
                              lossfun=F.sigmoid_cross_entropy,
                              accfun=F.binary_accuracy)

    print('Loading model parameter from ', model_path)
    serializers.load_npz(model_path, model)

    target_dataset = val
    target_smiles = val_smiles

    val_mols = [Chem.MolFromSmiles(smi) for smi in tqdm(val_smiles)]

    pyridine_mol = Chem.MolFromSmarts(PYRIDINE_SMILES)
    pyridine_index = np.where(np.array([mol.HasSubstructMatch(pyridine_mol) for mol in val_mols]) == True)
    val_pyridine_mols = np.array(val_mols)[pyridine_index]

    # It only extracts one substructure, not expected behavior
    # val_pyridine_pos = [set(mol.GetSubstructMatch(pi)) for mol in val_pyridine_mols]
    def flatten_tuple(x):
        return [element for tupl in x for element in tupl]

    val_pyridine_pos = [flatten_tuple(mol.GetSubstructMatches(pyridine_mol)) for mol in val_pyridine_mols]

    # print('pyridine_index', pyridine_index)
    # print('val_pyridine_mols', val_pyridine_mols.shape)
    # print('val_pyridine_pos', val_pyridine_pos)
    # print('val_pyridine_pos length', [len(k) for k in val_pyridine_pos])

    pyrigine_dataset = NumpyTupleDataset(*target_dataset.features[pyridine_index, :])
    pyrigine_smiles = target_smiles[pyridine_index]
    print('pyrigine_dataset', len(pyrigine_dataset), len(pyrigine_smiles))

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
        atom_embed = classifier.predictor.graph_conv.embed(atom)
        return atom_embed, adj, t

    def eval_fun(*inputs):
        atom_embed, adj, t = inputs
        prob = classifier.predictor(atom_embed, adj)
        out = F.sum(prob)
        return out

    calculator_method = args.calculator
    print('calculator method', calculator_method)
    if calculator_method == 'gradient':
        # option1: Gradient
        calculator = GradientCalculator(
            classifier, eval_fun=eval_fun,
            # target_key='embed', eval_key='out',
            target_key=0,
            # multiply_target=True  # this will calculate grad * input
        )
    elif calculator_method == 'integrated_gradients':
        # option2: IntegratedGradients
        calculator = IntegratedGradientsCalculator(
            classifier, eval_fun=eval_fun,
            # target_key='embed', eval_key='out',
            target_key=0, steps=10
        )
    elif calculator_method == 'occlusion':
        # option3: Occlusion
        def eval_fun_occlusion(*inputs):
            atom_embed, adj, t = inputs
            prob = classifier.predictor(atom_embed, adj)
            # Do not take sum, instead return batch-wise score
            out = F.sigmoid(prob)
            return out
        calculator = OcclusionCalculator(
            classifier, eval_fun=eval_fun_occlusion,
            # target_key='embed', eval_key='out',
            target_key=0, slide_axis=1
        )
    else:
        raise ValueError("[ERROR] Unexpected value calculator_method={}".format(calculator_method))

    M = 100
    num = 20
    rates = np.linspace(0.1, 1, num=num)
    print('M', M)

    # --- VanillaGrad ---
    saliency_arrays = calculator.compute_vanilla(
        pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun)
    saliency = calculator.transform(
        saliency_arrays, ch_axis=3, method='square')
    # saliency_arrays -> M, batch_size, max_atom, ch_dim
    # print('saliency_arrays', saliency_arrays.shape)
    # saliency -> batch_size, max_atom
    # print('saliency', saliency.shape)
    saliency_vanilla = clip_original_size(saliency, num_atoms)

    # recall & precision
    vanilla_recall, vanilla_precision = calc_recall_precision(saliency_vanilla, rates, val_pyridine_pos)
    print('vanilla_recall', vanilla_recall)
    print('vanilla_precision', vanilla_precision)

    # --- SmoothGrad ---
    saliency_arrays = calculator.compute_smooth(
        pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun,
        M=M,
        mode='absolute', scale=0.15  # previous implementation
        # mode='relative', scale=0.05
    )
    saliency = calculator.transform(
        saliency_arrays, ch_axis=3, method='square')

    saliency_smooth = clip_original_size(saliency, num_atoms)

    # recall & precision
    smooth_recall, smooth_precision = calc_recall_precision(saliency_smooth, rates, val_pyridine_pos)
    print('smooth_recall', smooth_recall)
    print('smooth_precision', smooth_precision)

    # --- BayesGrad ---
    # bayes grad is calculated by compute_vanilla with train=True
    saliency_arrays = calculator.compute_vanilla(
        pyrigine_dataset, converter=concat_mols, preprocess_fn=preprocess_fun,
        M=M, train=True)
    saliency = calculator.transform(
        saliency_arrays, ch_axis=3, method='square', lam=0)
    saliency_bayes = clip_original_size(saliency, num_atoms)

    bayes_recall, bayes_precision = calc_recall_precision(saliency_bayes, rates, val_pyridine_pos)
    print('bayes_recall', bayes_recall)
    print('bayes_precision', bayes_precision)

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(vanilla_recall, vanilla_precision, 'k-', color='blue', label='VanillaGrad')
    plt.plot(smooth_recall, smooth_precision, 'k-', color='green', label='SmoothGrad')
    plt.plot(bayes_recall, bayes_precision, 'k-', color='red', label='BayesGrad(Ours)')
    plt.axhline(y=vanilla_precision[-1], color='gray', linestyle='--')
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    if save_path:
        print('saved to ', save_path)
        plt.savefig(save_path)
        # plt.savefig('artificial_pr.eps')
    else:
        plt.show()


if __name__ == '__main__':
    # Disable errors by RDKit occurred in preprocessing Tox21 dataset.
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    # show INFO level log from chainer chemistry
    logging.basicConfig(level=logging.INFO)

    args = parse()
    # --- extracting configs ---
    dirpath = args.dirpath
    json_path = os.path.join(dirpath, 'args.json')
    if not os.path.exists(json_path):
        raise ValueError(
            'json_path {} not found! Execute train_tox21.py beforehand.'.format(json))
    with open(json_path, 'r') as f:
        train_args = json.load(f)

    method = train_args['method']
    labels = train_args['label']  # 'pyridine'

    unit_num = train_args['unit_num']
    conv_layers = train_args['conv_layers']
    class_num = 1
    n_layers = train_args['n_layers']
    dropout_ratio = train_args['dropout_ratio']
    num_train = train_args['num_train']
    # seed = train_args['seed']
    # --- extracting configs end ---

    model_path = os.path.join(dirpath, 'predictor.npz')
    save_path = os.path.join(
        dirpath, 'precision_recall_{}.png'.format(args.calculator))
    # --- config end ---

    main(method, labels, unit_num, conv_layers, class_num, n_layers,
         dropout_ratio, model_path, save_path)
