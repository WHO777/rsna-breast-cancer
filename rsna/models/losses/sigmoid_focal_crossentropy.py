import tensorflow_addons as tfa

from .builder import LOSSES


@LOSSES.register_module('SigmoidFocalCrossEntropy')
def sigmoid_focal_crossentropy(**kwargs):
    return tfa.losses.SigmoidFocalCrossEntropy(**kwargs)

