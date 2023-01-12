import tensorflow as tf

from .builder import LOSSES


@LOSSES.register_module("BinaryCrossEntropy")
def binary_crossentropy(**kwargs):
    return tf.keras.losses.BinaryCrossentropy(**kwargs)
