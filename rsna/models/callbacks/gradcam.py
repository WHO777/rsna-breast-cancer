import inspect
import functools
import logging
from pathlib import Path
from functools import partial

import tensorflow as tf
from tqdm import tqdm

from .builder import CALLBACKS
from rsna.utils import model as model_utils
from rsna.utils import gradcam as gradcam_utils


@CALLBACKS.register_module()
class GradCam(tf.keras.callbacks.Callback):

    def __init__(self, dataset, model_dir, last_conv_layer, alpha=0.4, pred_index=0):
        if isinstance(dataset, tf.keras.utils.Sequence):
            test_batch = dataset.__getitem__(0)
            with_labels = isinstance(test_batch, tuple) and len(test_batch) == 2
            if with_labels:
                output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=test_batch[0].dtype),
                                    tf.TensorSpec(shape=(None,), dtype=test_batch[1].dtype))
            else:
                output_signature = tf.TensorSpec(shape=(None, None, None, 3), dtype=test_batch.dtype)
            dataset = tf.data.Dataset.from_generator(partial(self.generator_fn, dataset=dataset),
                                                     output_signature=output_signature)
        else:
            element_spec = dataset.element_spec
            with_labels = isinstance(element_spec, tuple) and len(element_spec) == 2

        gradcam_dir = Path(model_dir) / 'gradcam'
        if not gradcam_dir.is_dir():
            gradcam_dir.mkdir(parents=True)

        self.dataset = dataset
        self.with_labels = with_labels
        self.gradcam_dir = gradcam_dir
        self.last_conv_layer = last_conv_layer
        self.alpha = alpha
        self.pred_index = pred_index

    def on_epoch_end(self, epoch, logs=None):
        grad_model = tf.keras.Model([self.model.inputs], [model_utils.get_layer_from_model(
            self.model, self.last_conv_layer, recursive=True).output, self.model.output])
        gradcam_fn = functools.partial(gradcam_utils.get_gradcam, grad_model=grad_model,
                                       alpha=self.alpha, pred_index=self.pred_index)
        epoch_dir = self.gradcam_dir / ('epoch_' + str(epoch))
        dataset = list(self.dataset.as_numpy_iterator())
        for i, data in enumerate(tqdm(dataset)):
            images, labels = data if self.with_labels else (data, None)
            images = tf.cast(images, tf.float32)
            map_kwargs = inspect.signature(tf.vectorized_map)
            gradcam_batch = tf.vectorized_map(gradcam_fn, images, warn=False) \
                if 'warn' in map_kwargs.parameters else tf.vectorized_map(gradcam_fn, images)
            for j, gradcam in enumerate(gradcam_batch):
                gradcam = tf.cast(gradcam, tf.uint8)
                label = labels[j] if labels is not None else None
                logging.info(epoch_dir)
                epoch_dir.mkdir(parents=True, exist_ok=True)
                filepath = epoch_dir / f'batch_{i}_id_{j}'
                filepath = str(filepath) + f'_label_{label}.jpg' if label else str(filepath) + '.jpg'
                tf.keras.utils.save_img(filepath, gradcam)
        logging.info(f'Writen to {epoch_dir}')

    @staticmethod
    def generator_fn(dataset):
        for i in range(dataset.__len__()):
            yield dataset.__getitem__(i)
