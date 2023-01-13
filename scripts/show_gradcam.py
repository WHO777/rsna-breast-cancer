from pathlib import Path

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import tensorflow as tf
import cv2
import keras_efficientnet_v2

from rsna.utils import gradcam
from rsna.utils import model as model_utils

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string('images_dir', None, '')
    flags.DEFINE_string('saved_model_dir', None, '')
    flags.DEFINE_string('last_conv_layer', 'top_conv', '')
    flags.DEFINE_string('output_dir', None, '')
    flags.DEFINE_float('alpha', 0.4, '')
    flags.DEFINE_integer('pred_index', 0, '')


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
    return image


def main(_):
    images_dir = Path(FLAGS.images_dir).absolute()
    output_dir = Path(FLAGS.output_dir).absolute()

    assert images_dir.is_dir()
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # model = tf.keras.models.load_model(FLAGS.saved_model_dir)
    # grad_model = tf.keras.Model([model.inputs], [model_utils.get_layer_from_model(
    #     model, FLAGS.last_conv_layer, recursive=True).output, model.output])

    image = tf.keras.layers.Input((1024, 1024, 3), name='image', dtype=tf.uint8)
    image_norm = normalize(image)
    outputs = keras_efficientnet_v2.EfficientNetV2T(
        input_shape=[1024, 1024, 3],
        pretrained=None,
        num_classes=1,
        classifier_activation='sigmoid',
        input_tensor=image_norm,
        dropout=0.30,
    )(image_norm)
    model = tf.keras.models.Model(inputs=image, outputs=outputs)
    model.load_weights('/app/test/model.h5')
    grad_model = tf.keras.Model([model.inputs], [model_utils.get_layer_from_model(
        model, FLAGS.last_conv_layer, recursive=True).output, model.output])
    from tensorflow.keras.utils import plot_model
    plot_model(grad_model, '/app/test/model.png')

    for image_path in tqdm(list(images_dir.iterdir())):
        image = cv2.imread(str(image_path))
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)
        gradcam_result = gradcam.get_gradcam(image, grad_model, FLAGS.alpha, FLAGS.pred_index)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), gradcam_result.numpy())


if __name__ == '__main__':
    define_flags()
    flags.mark_flag_as_required('images_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
