# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hparams for model architecture and trainer."""
import ast
from collections import abc
import copy
from typing import Any, Dict, Text
import six
import tensorflow as tf
import yaml


def eval_str_fn(val):
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


# pylint: disable=protected-access
class Config(object):
  """A config utility class."""

  def __init__(self, config_dict=None):
    self.update(config_dict)

  def __setattr__(self, k, v):
    self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __getitem__(self, k):
    return self.__dict__[k]

  def __repr__(self):
    return repr(self.as_dict())

  def __deepcopy__(self, memodict):
    return type(self)(self.as_dict())

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in six.iteritems(config_dict):
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def keys(self):
    return self.__dict__.keys()

  def override(self, config_dict_or_str, allow_new_keys=False):
    """Update members while disallowing new keys."""
    if isinstance(config_dict_or_str, str):
      if not config_dict_or_str:
        return
      elif '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path):
    """Write a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            if '*' in v:
              # we reserve * to split arrays.
              return {k: [eval_str_fn(vv) for vv in v.split('*')]}
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

        def merge_dict_recursive(target, src):
          """Recursively merge two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], abc.Mapping))):
              merge_dict_recursive(target[k], src[k])
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict
    # pylint: enable=protected-access


def get_default_config():
    h = Config()

    h.devices = [0, 1, 2]
    h.mixed_precision = 'float32'

    # model
    h.model = Config()
    h.model.backbone = Config()
    h.model.backbone.type = 'EfficientNetV2B3'
    h.model.backbone.weights = 'imagenet'
    h.model.backbone.freeze = False
    h.model.backbone.input_shape = (1024, 1024, 3)

    h.model.head = Config()
    h.model.head.type = 'SimpleHead'
    # h.model.head.num_hidden_units = 1000
    h.model.head.num_classes = 1
    # h.model.head.batch_norm = True
    # h.model.head.relu = False

    # optimizer
    h.optimizer = Config()
    h.optimizer.type = 'adam'
    h.optimizer.learning_rate = 0.0001

    # loss
    h.loss = Config()
    h.loss.type = 'SigmoidFocalCrossEntropy'
    h.loss.from_logits = False
    h.loss.alpha = 0.8
    h.loss.gamma = 2.0

    # ("TensorBoard", dict(log_dir='logs', profile_batch=(5, 110), histogram_freq=1)),
    # dict(type="GradCam", last_conv_layer='top_conv', alpha=0.5, pred_index=0),
    # callbacks
    h.callbacks = [
      ("ModelCheckpoint", dict(filepath='checkpoints', save_best_only=True, monitor='val_auc', mode='max')),
      ("TensorBoard", dict(log_dir='logs')),
      ("CSVLogger", dict(filename='history.csv'))
    ]

    # metrics
    h.metrics = [
        dict(type='pFBeta'),
        ('AUC', {}),
    ]

    # scheduler
    # h.scheduler = Config()
    # h.scheduler.type = 'CosineDecayRestarts'
    # h.scheduler.initial_learning_rate = 0.001
    # h.scheduler.first_decay_steps = 10

    # train
    h.train = Config()
    h.train.epochs = 10
    h.train.dataset = Config()
    h.train.dataset.type = 'CSVDataset'
    h.train.dataset.as_generator = True
    h.train.dataset.csv = '/app/dataset/train_fold_0.csv'
    h.train.dataset.images_column = 'image_path'
    h.train.dataset.labels_column = 'cancer'
    h.train.dataset.image_shape = (1024, 1024)
    h.train.dataset.shuffle = True
    h.train.dataset.batch_size = 12

    # val
    h.val = Config()
    h.val.dataset = Config()
    h.val.dataset.type = 'CSVDataset'
    h.val.dataset.as_generator = True
    h.val.dataset.csv = '/app/dataset/val_fold_0.csv'
    h.val.dataset.images_column = 'image_path'
    h.val.dataset.labels_column = 'cancer'
    h.val.dataset.image_shape = (1024, 1024)
    h.val.dataset.shuffle = False
    h.val.dataset.batch_size = 12

    # wandb
    # h.wandb = Config()
    # h.wandb.run_name = "metrics_only"

    return h

