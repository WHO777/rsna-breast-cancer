from pathlib import Path

import tensorflow as tf


def decode_image(image_path):
    file_bytes = tf.io.read_file(image_path)
    image_path = Path(str(image_path))
    if image_path.suffix == '.png':
        image = tf.image.decode_png(file_bytes, channels=3)
    elif image_path.suffix in ['.jpg', '.jpeg']:
        image = tf.image.decode_jpeg(file_bytes, channels=3)
    else:
        image = tf.image.decode_image(file_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image


def scale_image(image, scale_min=0.0, scale_max=255.0):
    image_dtype = image.dtype
    image = tf.cast(image, tf.float32)
    image_min = tf.reduce_min(image, axis=[0, 1])
    image_max = tf.reduce_max(image, axis=[0, 1])
    image = tf.math.divide_no_nan((image - image_min), (image_max - image_min))
    image = image * (scale_max - scale_min) + scale_min
    image = tf.cast(image, image_dtype)
    return image


def resize_and_pad(image, image_shape=(224, 224)):
    image_dtype = image.dtype
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    image_scale_y = tf.cast(image_shape[0], tf.float32) / height
    image_scale_x = tf.cast(image_shape[1], tf.float32) / width
    image_scale = tf.minimum(image_scale_x, image_scale_y)
    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    image = tf.image.resize(image, (scaled_height, scaled_width))
    image = tf.image.pad_to_bounding_box(image, 0, 0,
                                         image_shape[0],
                                         image_shape[1])
    image = tf.cast(image, image_dtype)
    return image
