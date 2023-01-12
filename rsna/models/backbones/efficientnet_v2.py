import tensorflow as tf

from .builder import BACKBONES


@BACKBONES.register_module('EfficientNetV2B0')
def efficient_net_v2_b0(**kwargs):
    return tf.keras.applications.efficientnet_v2.EfficientNetV2B0(**kwargs)


@BACKBONES.register_module('EfficientNetV2B1')
def efficient_net_v2_b1(**kwargs):
    return tf.keras.applications.efficientnet_v2.EfficientNetV2B0(**kwargs)


@BACKBONES.register_module('EfficientNetV2B2')
def efficient_net_v2_b2(**kwargs):
    return tf.keras.applications.efficientnet_v2.EfficientNetV2B0(**kwargs)


@BACKBONES.register_module('EfficientNetV2B3')
def efficient_net_v2_b3(**kwargs):
    return tf.keras.applications.efficientnet_v2.EfficientNetV2B0(**kwargs)


@BACKBONES.register_module('EfficientNetV2L')
def efficient_net_v2_l(**kwargs):
    return tf.keras.applications.efficientnet_v2.EfficientNetV2L(**kwargs)
