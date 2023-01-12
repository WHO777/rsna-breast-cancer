from tensorflow import keras

from .builder import BACKBONES


@BACKBONES.register_module()
def resnet50(**kwargs):
    return keras.applications.resnet.ResNet50(**kwargs)


@BACKBONES.register_module()
def resnet101(**kwargs):
    return keras.applications.resnet.ResNet101(**kwargs)


@BACKBONES.register_module()
def resnet150(**kwargs):
    return keras.applications.resnet.ResNet150(**kwargs)
