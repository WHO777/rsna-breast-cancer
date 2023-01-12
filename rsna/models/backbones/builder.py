from mmcv.utils.registry import Registry

BACKBONES = Registry('backbones')


def build_backbone(cfg):
    return BACKBONES.build(cfg)
