import torch
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.common import Granularity


@register_observer
class Observer(BaseObserver):
    TYPE = "minmax"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)

    def calc_minmax(self):
        if self.granularity == Granularity.CHANNELWISE:
            data = self.data_cache.get_data_for_calibration(Granularity.CHANNELWISE)
            if self.groups == -1:
                max_val = data.max(axis=1).values
                min_val = data.min(axis=1).values
            else:
                channels = data.shape[0]
                if channels % self.groups != 0:
                    print("warning: channels is not an integer multiple of groups, set groups as number of channels")
                    groups = channels
                else:
                    groups = self.groups
                data = data.reshape(groups, -1)
                max_val = data.max(axis=1).values
                min_val = data.min(axis=1).values
                groupsize = channels // groups
                max_val = max_val.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
                min_val = min_val.reshape(-1, 1).repeat(1, groupsize).reshape(-1)
        elif self.granularity == Granularity.LAYERWISE:
            data = self.data_cache.get_data_for_calibration(Granularity.LAYERWISE)
            min_val, max_val = data.min(), data.max()
        else:
            raise NotImplementedError
        self.data_cache.reset()
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
