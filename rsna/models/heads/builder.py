from mmcv.utils.registry import Registry

HEADS = Registry('heads')


def build_head(cfg):
    return HEADS.build(cfg)
