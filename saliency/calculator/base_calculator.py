import warnings
from abc import ABCMeta
from abc import abstractmethod

import chainer
from chainer import cuda
from chainer.dataset.convert import concat_examples, \
    _concat_arrays_with_padding
from chainer.iterators import SerialIterator

from future.utils import with_metaclass
import numpy


_sampling_axis = 0


def _to_tuple(x):
    if not isinstance(x, tuple):
        x = (x,)
    return x


def _to_variable(x):
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    return x


def _extract_numpy(x):
    if isinstance(x, chainer.Variable):
        x = x.data
    return cuda.to_cpu(x)


class BaseCalculator(with_metaclass(ABCMeta, object)):

    def __init__(self, model):
        self.model = model  # type: chainer.Chain
        self._device = cuda.get_device_from_array(*model.params()).id
        # print('device', self._device)

    def compute(
            self, data, M=1, method='vanilla', batchsize=16,
            converter=concat_examples, retain_inputs=False, preprocess_fn=None,
            postprocess_fn=None, train=False):
        method_dict = {
            'vanilla': self.compute_vanilla,
            'smooth': self.compute_smooth,
            'bayes': self.compute_bayes,
        }
        return method_dict[method](
            data, batchsize=batchsize, M=M, converter=converter,
            retain_inputs=retain_inputs, preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn, train=train)

    def compute_vanilla(self, data, batchsize=16, M=1,
                        converter=concat_examples, retain_inputs=False,
                        preprocess_fn=None, postprocess_fn=None, train=False):
        """VanillaGrad"""
        saliency_list = []
        for _ in range(M):
            with chainer.using_config('train', train):
                saliency = self._forward(
                    data, fn=self._compute_core, batchsize=batchsize,
                    converter=converter,
                    retain_inputs=retain_inputs, preprocess_fn=preprocess_fn,
                    postprocess_fn=postprocess_fn)
                saliency_list.append(cuda.to_cpu(saliency))
        return numpy.stack(saliency_list, axis=_sampling_axis)

    def compute_smooth(self, data, M=10, batchsize=16,
                       converter=concat_examples, retain_inputs=False,
                       preprocess_fn=None, postprocess_fn=None, train=False,
                       scale=0.15, mode='relative'):
        """SmoothGrad
        Reference
        https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L54
        """

        def smooth_fn(*inputs):
            #TODO: support cupy input
            target_array = inputs[self.target_key].data
            xp = cuda.get_array_module(target_array)

            noise = xp.random.normal(
                0, scale, inputs[self.target_key].data.shape)
            if mode == 'absolute':
                # `scale` is used as is
                pass
            elif mode == 'relative':
                # `scale_axis` is used to calculate `max` and `min` of target_array
                # As default, all axes except batch axis are treated as `scale_axis`.
                scale_axis = tuple(range(1, target_array.ndim))
                noise = noise * (xp.max(target_array, axis=scale_axis, keepdims=True)
                                 - xp.min(target_array, axis=scale_axis, keepdims=True))
                # print('[DEBUG] noise', noise.shape)
            else:
                raise ValueError("[ERROR] Unexpected value mode={}"
                                 .format(mode))
            inputs[self.target_key].data += noise
            return self._compute_core(*inputs)

        saliency_list = []
        for _ in range(M):
            with chainer.using_config('train', train):
                saliency = self._forward(
                    data, fn=smooth_fn, batchsize=batchsize,
                    converter=converter,
                    retain_inputs=retain_inputs, preprocess_fn=preprocess_fn,
                    postprocess_fn=postprocess_fn)
            saliency_array = cuda.to_cpu(saliency)
            saliency_list.append(saliency_array)
        return numpy.stack(saliency_list, axis=_sampling_axis)

    def compute_bayes(self, data, M=10, batchsize=16,
                      converter=concat_examples, retain_inputs=False,
                      preprocess_fn=None, postprocess_fn=None, train=True):
        """BayesGrad"""
        warnings.warn('`compute_bayes` method maybe deleted in the future...'
                      'please use `compute_vanilla` with train=True instead.')
        assert train
        # This is actually just an alias of `compute_vanilla` with `train=True`
        # Maybe deleted in the future.
        return self.compute_vanilla(
            data, M=M, batchsize=batchsize, converter=converter,
            retain_inputs=retain_inputs, preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn, train=True)

    def transform(self, saliency_arrays, method='raw', lam=0, ch_axis=2):
        if method == 'raw':
            h = numpy.sum(saliency_arrays, axis=ch_axis)
        elif method == 'abs':
            h = numpy.sum(numpy.abs(saliency_arrays), axis=ch_axis)
        elif method == 'square':
            h = numpy.sum(saliency_arrays ** 2, axis=ch_axis)
        else:
            raise ValueError('')

        sampling_axis = _sampling_axis
        if lam == 0:
            return numpy.mean(h, axis=sampling_axis)
        else:
            if h.shape[sampling_axis] == 1:
                # VanillaGrad does not support LCB/UCB calculation
                raise ValueError(
                    'saliency_arrays.shape[{}] must be larget than 1'.format(sampling_axis))
            return numpy.mean(h, axis=sampling_axis) + lam * numpy.std(
                h, axis=sampling_axis)

    @abstractmethod
    def _compute_core(self, *inputs):
        raise NotImplementedError

    def _forward(self, data, fn=None, batchsize=16,
                 converter=concat_examples, retain_inputs=False,
                 preprocess_fn=None, postprocess_fn=None):
        """Forward data by iterating with batch

        Args:
            data: "train_x array" or "chainer dataset"
            fn (Callable): Main function to forward. Its input argument is
                either Variable, cupy.ndarray or numpy.ndarray, and returns
                Variable.
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.

        Returns (tuple or numpy.ndarray): forward result

        """
        input_list = None
        output_list = None
        it = SerialIterator(data, batch_size=batchsize, repeat=False,
                            shuffle=False)
        for batch in it:
            inputs = converter(batch, self._device)
            inputs = _to_tuple(inputs)

            if preprocess_fn:
                inputs = preprocess_fn(*inputs)
                inputs = _to_tuple(inputs)

            inputs = (_to_variable(x) for x in inputs)

            outputs = fn(*inputs)

            # Init
            if retain_inputs:
                if input_list is None:
                    input_list = [[] for _ in range(len(inputs))]
                for j, input in enumerate(inputs):
                    input_list[j].append(cuda.to_cpu(input))
            if output_list is None:
                output_list = [[] for _ in range(len(outputs))]

            if postprocess_fn:
                outputs = postprocess_fn(*outputs)
                outputs = _to_tuple(outputs)
            for j, output in enumerate(outputs):
                output_list[j].append(_extract_numpy(output))

        if retain_inputs:
            self.inputs = [numpy.concatenate(
                in_array) for in_array in input_list]

        result = [_concat(output) for output in output_list]

        # result = [numpy.concatenate(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            return result


def _concat(batch_list):
    try:
        return numpy.concatenate(batch_list)
    except Exception as e:
        # Thre is a case that each input has different shape,
        # we cannot concatenate into array in this case.

        elem_list = [elem for batch in batch_list for elem in batch]
        return _concat_arrays_with_padding(elem_list, padding=0)
