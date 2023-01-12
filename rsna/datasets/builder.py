from mmcv.utils.registry import Registry

DATASETS = Registry('datasets')


def build_bataset(cfg):
    return DATASETS.build(cfg)
