import itertools

import numpy

import chainer
import six

from saliency.calculator.base_calculator import BaseCalculator


def _to_tuple(x):
    if isinstance(x, int):
        x = (x,)
    elif isinstance(x, (list, tuple)):
        x = tuple(x)
    else:
        raise TypeError('Unexpected type {}'.format(type(x)))
    return x


class OcclusionCalculator(BaseCalculator):

    def __init__(self, model, eval_fun=None, eval_key=None,
                 enable_backprop=False, size=1, slide_axis=(2, 3),
                 target_key=0):
        super(OcclusionCalculator, self).__init__(model)
        # self.model = model
        # self._device = cuda.get_array_module(model)
        self.eval_fun = eval_fun
        self.eval_key = eval_key
        self.enable_backprop = enable_backprop
        self.slide_axis = _to_tuple(slide_axis)
        size = _to_tuple(size)
        if len(self.slide_axis) != size:
            size = size * len(self.slide_axis)
        self.size = size
        print('slide_axis', self.slide_axis, 'size', self.size)
        self.target_key = target_key

    def _compute_core(self, *inputs):
        if self.target_key is None:
            target_var = inputs
        elif isinstance(self.target_key, int):
            target_var = inputs[self.target_key]
        else:
            raise TypeError('Unexpected type {} for target_key'
                            .format(type(self.target_key)))

        def _extract_score(result):
            if self.eval_key is None:
                score = result
            elif isinstance(self.eval_key, str):
                score = result[self.eval_key]
            else:
                raise TypeError('Unexpected type {} for eval_key'
                                .format(type(self.eval_key)))
            return score

        # Usually, backward() is not necessary for calculating occlusion
        with chainer.using_config('enable_backprop', self.enable_backprop):
            original_result = self.eval_fun(*inputs)
        original_score = _extract_score(original_result)

        # TODO: xp and value assign dynamically
        xp = numpy
        value = 0.

        # fill with `value`
        target_dim = target_var.ndim
        batch_size = target_var.shape[0]
        occlusion_window_shape = [1] * target_dim
        occlusion_window_shape[0] = batch_size
        for axis, size in zip(self.slide_axis, self.size):
            occlusion_window_shape[axis] = size
        occlusion_scores_shape = [1] * target_dim
        occlusion_scores_shape[0] = batch_size
        for axis, size in zip(self.slide_axis, self.size):
            occlusion_scores_shape[axis] = target_var.shape[axis]
        # print('[DEBUG] occlusion_shape', occlusion_window_shape)
        occlusion_window = xp.ones(occlusion_window_shape, dtype=target_var.dtype) * value
        # print('[DEBUG] occlusion_window.shape', occlusion_window.shape)
        occlusion_scores = xp.zeros(occlusion_scores_shape, dtype=xp.float32)
        # print('[DEBUG] occlusion_scores.shape', occlusion_scores.shape)

        def _extract_index(slide_axis, size, start_indices):
            colon = slice(None)
            index = [colon] * target_dim
            for axis, size, start in zip(slide_axis, size, start_indices):
                index[axis] = slice(start, start + size, 1)
            return index

        end_list = [target_var.data.shape[axis] - size for axis, size
                    in zip(self.slide_axis, self.size)]

        for start in itertools.product(*[six.moves.range(end) for end in end_list]):
            target_var_occluded = target_var.data.copy()
            occlude_index = _extract_index(self.slide_axis, self.size, start)
            target_var_occluded[occlude_index] = occlusion_window

            # Usually, backward() is not necessary for calculating occlusion
            with chainer.using_config('enable_backprop', self.enable_backprop):
                # TODO: consider case target_key=None
                occluded_inputs = (target_var_occluded if self.target_key == i else elem
                                   for i, elem in enumerate(inputs))
                occluded_result = self.eval_fun(*occluded_inputs)
            occluded_score = _extract_score(occluded_result)
            score_diff_var = original_score - occluded_score
            # TODO: expand_dim dynamically
            score_diff = xp.broadcast_to(score_diff_var.data[:, :, None], occlusion_window_shape)
            occlusion_scores[occlude_index] += score_diff
        outputs = (occlusion_scores,)
        return outputs


if __name__ == '__main__':
    # TODO: test
    raise NotImplementedError()
    oc = OcclusionCalculator(model)
    saliency_array = oc.compute_vanilla()
    saliency = oc.transform(saliency_array)
