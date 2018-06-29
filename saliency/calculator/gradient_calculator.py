from saliency.calculator.base_calculator import BaseCalculator


class GradientCalculator(BaseCalculator):

    def __init__(self, model, eval_fun=None, eval_key=None, target_key=0,
                 multiply_target=False):
        super(GradientCalculator, self).__init__(model)
        # self.model = model
        # self._device = cuda.get_array_module(model)
        self.eval_fun = eval_fun
        self.eval_key = eval_key
        self.target_key = target_key

        self.multiply_target = multiply_target

    def _compute_core(self, *inputs):
        # outputs = fn(*inputs)
        # outputs = _to_tuple(outputs)
        self.model.cleargrads()
        result = self.eval_fun(*inputs)
        if self.eval_key is None:
            eval_var = result
        elif isinstance(self.eval_key, str):
            eval_var = result[self.eval_key]
        else:
            raise TypeError('Unexpected type {} for eval_key'
                            .format(type(self.eval_key)))
        # TODO: Consider how deal with the case when eval_var is not scalar,
        # 1. take sum
        # 2. raise error (default behavior)
        # I think option 1 "take sum" is better, since gradient is calculated
        # automatically independently in that case.
        eval_var.backward(retain_grad=True)

        if self.target_key is None:
            target_var = inputs
        elif isinstance(self.target_key, int):
            target_var = inputs[self.target_key]
        else:
            raise TypeError('Unexpected type {} for target_key'
                            .format(type(self.target_key)))
        saliency = target_var.grad
        if self.multiply_target:
            saliency *= target_var.data
        outputs = (saliency,)
        return outputs
