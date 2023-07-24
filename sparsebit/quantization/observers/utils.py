from sparsebit.quantization.common import Granularity

def mse_loss(pred, tgt, granularity:Granularity):
    if granularity == Granularity.CHANNELWISE:
        return ((pred - tgt) ** 2).mean(-1)
    elif granularity == Granularity.LAYERWISE:
        return ((pred - tgt) ** 2).mean()
    else:
        raise NotImplementedError
