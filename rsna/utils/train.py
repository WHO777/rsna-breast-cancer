import tensorflow as tf


class DeviceSpec:
    def __init__(self, gpu):
        self.gpu = gpu
        self._index = int(gpu.name[-1])
        self._name = ':'.join(gpu.name.split(':')[-2:])

    @property
    def index(self):
        return self._index

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @index.setter
    def index(self, value):
        self._index = value

    def __repr__(self):
        return str(dict(name=self._name, index=self._index))


def get_strategy(device_indexes=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device_specs = [DeviceSpec(gpu) for gpu in gpus]
    device_indexes = device_indexes or [spec.index for spec in device_specs]
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(
            [device.name for device in device_specs if device.index in device_indexes])
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
    return strategy


def set_precision_policy(policy_name=None):
    if not policy_name:
        return

    assert policy_name in ('float32', 'mixed_float16', 'mixed_bfloat32')
    policy = tf.keras.mixed_precision.Policy(policy_name)
    tf.keras.mixed_precision.set_global_policy(policy)
