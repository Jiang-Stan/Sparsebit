import torch
from sparsebit.quantization.quantizers import Quantizer as BaseQuantizer
from sparsebit.quantization.quantizers import register_quantizer
from .quant_tensor import STE


@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "uniform"

    def __init__(self, config):
        super(Quantizer, self).__init__(config)

    def _forward(self, x_f, scale, zero_point):
        if hasattr(self.observer, "perm"):
            perm_weight_f = x_f[:,self.observer.perm]
            perm_weight_dq = STE.apply(perm_weight_f, scale, zero_point, self.qdesc, self.backend)
            x_dq = perm_weight_dq[:, self.observer.inv_perm]
        else:
            x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        return x_dq
