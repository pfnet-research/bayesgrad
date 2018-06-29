import chainer
from chainer import functions as F
import chainer.links as L
import sys
import os

from chainer_chemistry.models import GGNN
from chainer_chemistry.models import NFP
from chainer_chemistry.models import SchNet
from chainer_chemistry.models import WeaveNet

sys.path.append(os.path.dirname(__file__))
from models.nfp_drop import NFPDrop
from models.ggnn_drop import GGNNDrop


class MLPDrop(chainer.Chain):
    """Basic implementation for MLP with dropout"""
    # def __init__(self, hidden_dim, out_dim, n_layers=2, activation=F.relu):
    def __init__(self, out_dim, hidden_dim, n_layers=1, activation=F.relu,
                 dropout_ratio=0.25):
        super(MLPDrop, self).__init__()
        if n_layers <= 0:
            raise ValueError('n_layers must be positive integer, but set {}'
                             .format(n_layers))
        layers = [L.Linear(None, hidden_dim) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = L.Linear(None, out_dim)
        self.activation = activation
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = F.dropout(x, ratio=self.dropout_ratio)
        for l in self.layers:
            h = F.dropout(self.activation(l(h)), ratio=self.dropout_ratio)
        h = self.l_out(h)
        return h


def build_predictor(method, n_unit, conv_layers, class_num,
                    dropout_ratio=0.25, n_layers=1):
    print('dropout_ratio, n_layers', dropout_ratio, n_layers)
    mlp_class = MLPDrop
    if method == 'nfp':
        print('Use NFP predictor...')
        predictor = GraphConvPredictor(
            NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            mlp_class(out_dim=class_num, hidden_dim=n_unit, dropout_ratio=dropout_ratio,
                      n_layers=n_layers))
    elif method == 'nfpdrop':
        print('Use NFPDrop predictor...')
        predictor = GraphConvPredictor(
            NFPDrop(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers,
                    dropout_ratio=dropout_ratio),
            mlp_class(out_dim=class_num, hidden_dim=n_unit,
                      dropout_ratio=dropout_ratio,
                      n_layers=n_layers))
    elif method == 'ggnn':
        print('Use GGNN predictor...')
        predictor = GraphConvPredictor(
            GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            mlp_class(out_dim=class_num, hidden_dim=n_unit,
                      dropout_ratio=dropout_ratio, n_layers=n_layers))
    elif method == 'ggnndrop':
        print('Use GGNNDrop predictor...')
        predictor = GraphConvPredictor(
            GGNNDrop(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers,
                     dropout_ratio=dropout_ratio),
            mlp_class(out_dim=class_num, hidden_dim=n_unit,
                      dropout_ratio=dropout_ratio, n_layers=n_layers))
    elif method == 'schnet':
        print('Use SchNet predictor...')
        predictor = SchNet(out_dim=class_num, hidden_dim=n_unit,
                           n_layers=conv_layers, readout_hidden_dim=n_unit)
    elif method == 'weavenet':
        print('Use WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        predictor = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            mlp_class(out_dim=class_num, hidden_dim=n_unit,
                      dropout_ratio=dropout_ratio, n_layers=n_layers))
    else:
        raise ValueError('[ERROR] Invalid predictor: method={}'.format(method))
    return predictor


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv, mlp):
        """Constructor

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        x = self.mlp(x)
        return x

    def predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)
