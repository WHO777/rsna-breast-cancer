import tensorflow as tf

from .builder import OPTIMIZERS


@OPTIMIZERS.register_module()
def adam(**kwargs):
    return tf.keras.optimizers.Adam(**kwargs)
