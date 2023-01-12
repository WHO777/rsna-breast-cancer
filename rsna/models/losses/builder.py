from mmcv.utils.registry import Registry

LOSSES = Registry('losses')


def build_loss(cfg):
    return LOSSES.build(cfg)
