from functools import partial

import tensorflow as tf
import pandas as pd

from .builder import DATASETS
from utils.image import decode_image, resize_and_pad


def image_decoder(with_labels=True):
    def decode_image_fn(image_path, image_shape=(1024, 1024)):
        image = decode_image(image_path)
        image = resize_and_pad(image, image_shape)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.cast(image, tf.float32) / 255.
        return image

    def decode_image_with_labels(image_path, label, image_shape=(1024, 1024)):
        label = tf.cast(label, tf.float32)
        return decode_image_fn(image_path, image_shape), label

    return decode_image_with_labels if with_labels else decode_image_fn


def image_preprocessor(with_labels):
    def preprocess(image, augmentations=None):
        if augmentations is not None:
            for augmentation in augmentations:
                image = augmentation(image)
        return image

    def preprocess_with_labels(image, label, augmentations=None):
        return preprocess(image, augmentations), label

    return preprocess_with_labels if with_labels else preprocess


def image_decoder_and_preprocessor(with_labels):
    decoder = image_decoder(with_labels)
    preprocessor = image_preprocessor(with_labels)

    def image_decode_and_preprocess(image_path, image_shape=(1024, 1024), augmentations=None):
        return preprocessor(decoder(image_path, image_shape), augmentations)

    def image_decode_and_preprocess_with_labels(image_path, label, image_shape=(1024, 1024), augmentations=None):
        return preprocessor(*decoder(image_path, label, image_shape), augmentations)

    return image_decode_and_preprocess_with_labels if with_labels else image_decode_and_preprocess


class ImagesDataGen(tf.keras.utils.Sequence):

    def __init__(self,
                 image_paths,
                 decode_and_preprocess_fn,
                 labels=None,
                 batch_size=64,
                 shuffle=True,
                 ):
        self.image_paths = tf.convert_to_tensor(image_paths)
        self.labels = tf.convert_to_tensor(labels, dtype=tf.int32) if labels is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.decode_and_preprocess_fn = (lambda x: decode_and_preprocess_fn(*x)) \
            if labels is not None else decode_and_preprocess_fn
        self.num_samples = tf.shape(image_paths)[0]

    def __len__(self):
        return tf.cast(tf.math.ceil(self.num_samples / self.batch_size), tf.int32)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.labels is not None:
            labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
            fn_output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                                   tf.TensorSpec(shape=(), dtype=tf.float32))
            images, labels = tf.map_fn(self.decode_and_preprocess_fn, (image_paths, labels),
                                       fn_output_signature=fn_output_signature)
            return images, labels
        else:
            fn_output_signature = tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
            images = tf.map_fn(self.decode_and_preprocess_fn, image_paths, fn_output_signature=fn_output_signature)
            return images

    def on_epoch_end(self):
        if self.shuffle:
            shuffle_idxs = tf.range(self.num_samples)
            self.image_paths = tf.gather(self.image_paths, shuffle_idxs)
            if self.labels is not None:
                self.labels = tf.gather(self.labels, shuffle_idxs)


def get_heavy_dataset(image_paths,
                      decode_and_preprocess_fn,
                      labels=None,
                      batch_size=64,
                      shuffle=True,
                      ):
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    options = tf.data.Options()
    options.deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    dataset = dataset.shuffle(8 * batch_size, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.map(decode_and_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


@DATASETS.register_module('CSVDataset')
def from_csv(csv,
             images_column,
             labels_column=None,
             as_generator=False,
             image_shape=(640, 640),
             batch_size=64,
             shuffle=True,
             augmentations=None
             ):
    df = pd.read_csv(csv)
    image_paths = df[images_column].values
    labels = df[labels_column].values if labels_column is not None else None
    # import numpy as np
    # labels = np.array(labels)
    # image_paths = np.array(image_paths)
    # pos_labels = labels[labels == 1]
    # neg_labels = labels[labels != 1]
    # num_pos = pos_labels.shape[0]
    # pos_image_paths = image_paths[labels == 1]
    # neg_image_paths = image_paths[labels != 1]
    # image_paths = pos_image_paths[:num_pos].tolist() + neg_image_paths[:num_pos].tolist()
    # labels = pos_labels[:num_pos].tolist() + neg_labels[:num_pos].tolist()
    decode_and_preprocess_fn = partial(image_decoder_and_preprocessor(labels_column is not None),
                                       image_shape=image_shape, augmentations=augmentations)
    if as_generator:
        dataset = ImagesDataGen(image_paths, decode_and_preprocess_fn, labels, batch_size, shuffle)
    else:
        dataset = get_heavy_dataset(image_paths, decode_and_preprocess_fn, labels, batch_size, shuffle)
    return dataset
