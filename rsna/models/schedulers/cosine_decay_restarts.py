import tensorflow as tf

from .builder import SCHEDULERS


@SCHEDULERS.register_module("CosineDecayRestarts")
def cosine_decay_restarts(initial_learning_rate, first_decay_steps, **kwargs):
    return tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate,
                                                             first_decay_steps,
                                                             **kwargs)
