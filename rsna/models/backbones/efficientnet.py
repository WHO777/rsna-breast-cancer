from tensorflow import keras

from .builder import BACKBONES


@BACKBONES.register_module('EfficientNetB0')
def efficientnet_b0(**kwargs):
    return keras.applications.efficientnet.EfficientNetB0(**kwargs)


@BACKBONES.register_module('EfficientNetB1')
def efficientnet_b1(**kwargs):
    return keras.applications.efficientnet.EfficientNetB1(**kwargs)


@BACKBONES.register_module('EfficientNetB2')
def efficientnet_b2(**kwargs):
    return keras.applications.efficientnet.EfficientNetB2(**kwargs)


@BACKBONES.register_module('EfficientNetB3')
def efficientnet_b3(**kwargs):
    return keras.applications.efficientnet.EfficientNetB3(**kwargs)
