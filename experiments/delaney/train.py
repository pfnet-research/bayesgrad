from __future__ import print_function
import argparse
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
mpl.use('Agg')

from chainer import functions as F, cuda, Variable
from chainer import iterators as I
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E
from chainer.datasets import SubDataset
from chainer import serializers

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.models.prediction import Regressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import predictor
from saliency.calculator.gradient_calculator import GradientCalculator
from plot import plot_result


def save_result(dataset, model, dir_path, M):
    regressor = Regressor(model, lossfun=F.mean_squared_error)
    # model.to_cpu()

    def preprocess_fun(*inputs):
        atom, adj, t = inputs
        # HACKING for now...
        atom_embed = regressor.predictor.graph_conv.embed(atom)
        return atom_embed, adj, t

    def eval_fun(*inputs):
        atom_embed, adj, t = inputs
        prob = regressor.predictor(atom_embed, adj)
        out = F.sum(prob)
        return out

    gradient_calculator = GradientCalculator(
        regressor, eval_fun=eval_fun,
        target_key=0, multiply_target=True
    )

    def clip_original_size(saliency_, num_atoms_):
        """`saliency` array is 0 padded, this method align to have original
        molecule's length
        """
        assert len(saliency_) == len(num_atoms_)
        saliency_list = []
        for i in range(len(saliency_)):
            saliency_list.append(saliency_[i, :num_atoms_[i]])
        return saliency_list

    atoms = dataset.features[:, 0]
    num_atoms = [len(a) for a in atoms]

    print('calculating saliency... M={}'.format(M))
    # --- VanillaGrad ---
    saliency_arrays = gradient_calculator.compute_vanilla(
        dataset, converter=concat_mols, preprocess_fn=preprocess_fun)
    saliency = gradient_calculator.transform(
        saliency_arrays, ch_axis=3, method='raw')
    saliency_vanilla = clip_original_size(saliency, num_atoms)
    np.save(os.path.join(dir_path, "saliency_vanilla"), saliency_vanilla)

    # --- SmoothGrad ---
    saliency_arrays = gradient_calculator.compute_smooth(
        dataset, converter=concat_mols, preprocess_fn=preprocess_fun, M=M)
    saliency = gradient_calculator.transform(
        saliency_arrays, ch_axis=3, method='raw')
    saliency_smooth = clip_original_size(saliency, num_atoms)
    np.save(os.path.join(dir_path, "saliency_smooth"), saliency_smooth)

    # --- BayesGrad ---
    # train=True corresponds to BayesGrad
    saliency_arrays = gradient_calculator.compute_vanilla(
        dataset, converter=concat_mols, preprocess_fn=preprocess_fun, M=M,
        train=True)
    saliency = gradient_calculator.transform(
        saliency_arrays, ch_axis=3, method='raw', lam=0)
    saliency_bayes = clip_original_size(saliency, num_atoms)
    np.save(os.path.join(dir_path, "saliency_bayes"), saliency_bayes)


def get_dir_path(batchsize, n_unit, conv_layers, M, method):
    dir_path = "results/{}_M{}_conv{}_unit{}_b{}".format(method, M, conv_layers, n_unit, batchsize)
    dir_path = os.path.join("./", dir_path)
    return dir_path


def train(gpu, method, epoch, batchsize, n_unit, conv_layers, dataset, smiles, M, n_split, split_idx, order):
    n = len(dataset)
    assert len(order) == n
    left_idx = (n // n_split) * split_idx
    is_right_most_split = (n_split == split_idx + 1)
    if is_right_most_split:
        test_order = order[left_idx:]
        train_order = order[:left_idx]
    else:
        right_idx = (n // n_split) * (split_idx + 1)
        test_order = order[left_idx:right_idx]
        train_order = np.concatenate([order[:left_idx], order[right_idx:]])

    new_order = np.concatenate([train_order, test_order])
    n_train = len(train_order)

    # Standard Scaler for labels
    ss = StandardScaler()
    labels = dataset.get_datasets()[-1]
    train_label = labels[new_order[:n_train]]
    ss = ss.fit(train_label)  # fit only by train
    labels = ss.transform(dataset.get_datasets()[-1])
    dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1] + (labels,)))

    dataset_train = SubDataset(dataset, 0, n_train, new_order)
    dataset_test = SubDataset(dataset, n_train, n, new_order)

    # Network
    model = predictor.build_predictor(
       method, n_unit, conv_layers, 1, dropout_ratio=0.25, n_layers=1)

    train_iter = I.SerialIterator(dataset_train, batchsize)
    val_iter = I.SerialIterator(dataset_test, batchsize, repeat=False, shuffle=False)

    def scaled_abs_error(x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        scaled_x0 = ss.inverse_transform(cuda.to_cpu(x0))
        scaled_x1 = ss.inverse_transform(cuda.to_cpu(x1))
        diff = scaled_x0 - scaled_x1
        return np.mean(np.absolute(diff), axis=0)[0]

    regressor = Regressor(
        model, lossfun=F.mean_squared_error,
        metrics_fun={'abs_error': scaled_abs_error}, device=gpu)

    optimizer = O.Adam(alpha=0.0005)
    optimizer.setup(regressor)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu, converter=concat_mols)

    dir_path = get_dir_path(batchsize, n_unit, conv_layers, M, method)
    dir_path = os.path.join(dir_path, str(split_idx) + "-" + str(n_split))
    os.makedirs(dir_path, exist_ok=True)
    print('creating ', dir_path)
    np.save(os.path.join(dir_path, "test_idx"), np.array(test_order))

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=dir_path)
    trainer.extend(E.Evaluator(val_iter, regressor, device=gpu,
                               converter=concat_mols))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
                                  'validation/main/loss',
                                  'validation/main/abs_error',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # --- Plot regression evaluation result ---
    dataset_test = SubDataset(dataset, n_train, n, new_order)
    batch_all = concat_mols(dataset_test, device=gpu)
    serializers.save_npz(os.path.join(dir_path, "model.npz"), model)
    result = model(batch_all[0], batch_all[1])
    result = ss.inverse_transform(cuda.to_cpu(result.data))
    answer = ss.inverse_transform(cuda.to_cpu(batch_all[2]))
    plot_result(result, answer, save_filepath=os.path.join(dir_path, "result.png"))

    # --- Plot regression evaluation result end ---
    np.save(os.path.join(dir_path, "output.npy"), result)
    np.save(os.path.join(dir_path, "answer.npy"), answer)
    smiles_part = np.array(smiles)[test_order]
    np.save(os.path.join(dir_path, "smiles.npy"), smiles_part)

    # calculate saliency and save it.
    save_result(dataset, model, dir_path, M)


def main():
    # Supported preprocessing/network list
    parser = argparse.ArgumentParser(
        description='Regression with own dataset.')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--method', type=str, default='nfpdrop',
                        choices=['nfpdrop', 'ggnndrop', 'nfp', 'ggnn'])
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--layer', '-n', type=int, default=3)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--m', '-m', type=int, default=30)
    args = parser.parse_args()

    dataset_name = 'delaney'
    # labels = "measured log solubility in mols per litre"
    labels = None

    # Dataset preparation
    print('Preprocessing dataset...')
    method = args.method
    if 'nfp' in method:
        preprocess_method = 'nfp'
    elif 'ggnn' in method:
        preprocess_method = 'ggnn'
    else:
        raise ValueError('Unexpected method', method)
    preprocessor = preprocess_method_dict[preprocess_method]()
    data = get_molnet_dataset(
        dataset_name, preprocessor, labels=labels, return_smiles=True,
        frac_train=1.0, frac_valid=0.0, frac_test=0.0)
    dataset = data['dataset'][0]
    smiles = data['smiles'][0]

    epoch = args.epoch
    gpu = args.gpu

    n_unit_list = [32]
    random_state = np.random.RandomState(args.seed)
    n = len(dataset)

    M = args.m
    order = np.arange(n)
    random_state.shuffle(order)
    batchsize = args.batchsize
    for n_unit in n_unit_list:
        n_layer = args.layer
        n_split = 5
        for idx in range(n_split):
            print('Start training: ', idx+1, "/", n_split)
            dir_path = get_dir_path(batchsize, n_unit, n_layer, M, method)
            os.makedirs(dir_path, exist_ok=True)
            np.save(os.path.join(dir_path, "smiles.npy"), np.array(smiles))
            train(gpu, method, epoch, batchsize, n_unit, n_layer, dataset, smiles, M, n_split, idx, order)


if __name__ == '__main__':
    main()
