import tensorflow as tf

from .builder import LOSSES


@LOSSES.register_module("BinaryFocalCrossentropy")
def binary_focal_crossentropy(**kwargs):
    return tf.keras.losses.BinaryFocalCrossentropy(**kwargs)
