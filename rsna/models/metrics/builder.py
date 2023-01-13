import inspect

from tensorflow.keras.metrics import *
from mmcv.utils.registry import Registry

from rsna.utils.config import parse_objects_from_cfg

METRICS = Registry('metrics')


def build_metrics(cfg):
    metrics = []
    for metric in cfg:
        if isinstance(metric, dict):
            metrics.append(METRICS.build(metric))
        else:
            exec(inspect.getsource(parse_objects_from_cfg))
            new_locals = locals()
            exec("metric = parse_objects_from_cfg([metric], 'metrics')", new_locals)
            metric = new_locals['metric']
            metrics.append(metric)
    return metrics
