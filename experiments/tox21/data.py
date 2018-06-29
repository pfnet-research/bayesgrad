import os

import numpy
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from rdkit import Chem
from tqdm import tqdm

import utils


class _CacheNamePolicy(object):

    train_file_name = 'train.npz'
    val_file_name = 'val.npz'
    test_file_name = 'test.npz'
    smiles_file_name = 'smiles.npz'

    def _get_cache_directory_path(self, method, labels, prefix):
        if labels:
            return os.path.join(prefix, '{}_{}'.format(method, labels))
        else:
            return os.path.join(prefix, '{}_all'.format(method))

    def __init__(self, method, labels, prefix='input'):
        self.method = method
        self.labels = labels
        self.prefix = prefix
        self.cache_dir = self._get_cache_directory_path(method, labels, prefix)

    def get_train_file_path(self):
        return os.path.join(self.cache_dir, self.train_file_name)

    def get_val_file_path(self):
        return os.path.join(self.cache_dir, self.val_file_name)

    def get_test_file_path(self):
        return os.path.join(self.cache_dir, self.test_file_name)

    def get_smiles_path(self):
        return os.path.join(self.cache_dir, self.smiles_file_name)

    def create_cache_directory(self):
        try:
            os.makedirs(self.cache_dir)
        except OSError:
            if not os.path.isdir(self.cache_dir):
                raise


PYRIDINE_SMILES = 'c1ccncc1'


def hassubst(mol, smart=PYRIDINE_SMILES):
    return numpy.array(int(mol.HasSubstructMatch(Chem.MolFromSmarts(smart)))).astype('int32')


def load_dataset(method, labels, prefix='input'):
    method = 'nfp' if 'nfp' in method else method  # to deal with nfpdrop
    method = 'ggnn' if 'ggnn' in method else method  # to deal with ggnndrop
    policy = _CacheNamePolicy(method, labels, prefix)
    train_path = policy.get_train_file_path()
    val_path = policy.get_val_file_path()
    test_path = policy.get_test_file_path()
    smiles_path = policy.get_smiles_path()

    train, val, test = None, None, None
    train_smiles, val_smiles, test_smiles = None, None, None
    print()
    if os.path.exists(policy.cache_dir):
        print('load from cache {}'.format(policy.cache_dir))
        train = NumpyTupleDataset.load(train_path)
        val = NumpyTupleDataset.load(val_path)
        test = NumpyTupleDataset.load(test_path)
        train_smiles, val_smiles, test_smiles = utils.load_npz(smiles_path)
    if train is None or val is None or test is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        if labels == 'pyridine':
            train, val, test, train_smiles, val_smiles, test_smiles = D.get_tox21(
                preprocessor, labels=None, return_smiles=True)
            print('converting label into pyridine...')
            # --- Pyridine = 1 ---
            train_pyridine_label = [
                hassubst(Chem.MolFromSmiles(smi), smart=PYRIDINE_SMILES) for smi in tqdm(train_smiles)]
            val_pyridine_label = [
                hassubst(Chem.MolFromSmiles(smi), smart=PYRIDINE_SMILES) for smi in tqdm(val_smiles)]
            test_pyridine_label = [
                hassubst(Chem.MolFromSmiles(smi), smart=PYRIDINE_SMILES) for smi in tqdm(test_smiles)]

            train_pyridine_label = numpy.array(train_pyridine_label)[:, None]
            val_pyridine_label = numpy.array(val_pyridine_label)[:, None]
            test_pyridine_label = numpy.array(test_pyridine_label)[:, None]
            print('train positive/negative', numpy.sum(train_pyridine_label == 1), numpy.sum(train_pyridine_label == 0))
            train = NumpyTupleDataset(*train.features[:, :-1], train_pyridine_label)
            val = NumpyTupleDataset(*val.features[:, :-1], val_pyridine_label)
            test = NumpyTupleDataset(*test.features[:, :-1], test_pyridine_label)
        else:
            train, val, test, train_smiles, val_smiles, test_smiles = D.get_tox21(
                preprocessor, labels=labels, return_smiles=True)

        # Cache dataset
        policy.create_cache_directory()
        NumpyTupleDataset.save(train_path, train)
        NumpyTupleDataset.save(val_path, val)
        NumpyTupleDataset.save(test_path, test)
        train_smiles = numpy.array(train_smiles)
        val_smiles = numpy.array(val_smiles)
        test_smiles = numpy.array(test_smiles)
        utils.save_npz(smiles_path, (train_smiles, val_smiles, test_smiles))
    return train, val, test, train_smiles, val_smiles, test_smiles
