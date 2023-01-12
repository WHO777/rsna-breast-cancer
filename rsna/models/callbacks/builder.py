import inspect

from tensorflow.keras.callbacks import *
from mmcv.utils.registry import Registry

from rsna.utils import config as config_utils

CALLBACKS = Registry('callbacks')


def build_callbacks(cfg):
    callbacks = []
    for callback in cfg:
        if isinstance(callback, dict):
            callbacks.append(CALLBACKS.build(callback))
        else:
            exec(inspect.getsource(config_utils.parse_objects_from_cfg))
            new_locals = locals()
            exec("callback = parse_objects_from_cfg([callback], 'callbacks')[0]", new_locals)
            callback = new_locals['callback']
            callbacks.append(callback)
    return callbacks
