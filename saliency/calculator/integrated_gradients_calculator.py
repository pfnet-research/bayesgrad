import numpy
from chainer import Variable

from saliency.calculator.base_calculator import BaseCalculator
from saliency.calculator.gradient_calculator import GradientCalculator


class IntegratedGradientsCalculator(GradientCalculator):

    def __init__(self, model, eval_fun=None, eval_key=None, target_key=0,
                 baseline=None, steps=25):
        super(IntegratedGradientsCalculator, self).__init__(
            model, eval_fun=eval_fun, eval_key=eval_key, target_key=target_key,
            multiply_target=False
        )
        # self.target_key = target_key
        self.baseline = baseline or 0.
        self.steps = steps

    def get_target_var(self, inputs):
        if self.target_key is None:
            target_var = inputs
        elif isinstance(self.target_key, int):
            target_var = inputs[self.target_key]
        else:
            raise TypeError('Unexpected type {} for target_key'
                            .format(type(self.target_key)))
        return target_var

    def _compute_core(self, *inputs):

        total_grads = 0.
        target_var = self.get_target_var(inputs)
        base = self.baseline
        diff = target_var.data - base
        for alpha in numpy.linspace(0., 1., self.steps):
            # TODO: consider case target_key=None
            interpolated_inputs = (
                Variable(base + alpha * diff) if self.target_key == i else elem
                for i, elem in enumerate(inputs))
            total_grads += super(
                IntegratedGradientsCalculator, self)._compute_core(
                *interpolated_inputs)[0]
        saliency = total_grads * diff
        return saliency,
