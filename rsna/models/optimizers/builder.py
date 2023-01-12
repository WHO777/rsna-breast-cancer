from mmcv.utils.registry import Registry

OPTIMIZERS = Registry('optimizers')


def build_optimizer(cfg):
    return OPTIMIZERS.build(cfg)
