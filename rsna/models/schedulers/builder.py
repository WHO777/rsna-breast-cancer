from mmcv.utils.registry import Registry

SCHEDULERS = Registry('schedulers')


def build_scheduler(cfg):
    return SCHEDULERS.build(cfg)
