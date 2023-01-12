import argparse
from pathlib import Path

import tensorflow as tf
import keras_efficientnet_v2


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('saved_model_dir')
    args = parser.parse_args()
    return args


def get_another_model():
    image = tf.keras.layers.Input((1024, 1024, 3), name='image', dtype=tf.uint8)
    image_norm = normalize(image)
    outputs = keras_efficientnet_v2.EfficientNetV2T(
        input_shape=[1024, 1024, 3],
        pretrained=None,
        num_classes=1,
        classifier_activation='sigmoid',
        dropout=0.30,
    )(image_norm)
    model = tf.keras.models.Model(inputs=image, outputs=outputs)
    model.load_weights('/app/test/model.h5')
    return model


def main():
    args = parse_args()

    my_model = tf.keras.models.load_model(args.saved_model_dir)
    another_model = get_another_model()

    sources = list(Path(args.source).iterdir())
    for source in sources:
        bytes = tf.io.read_file(str(source))
        image = tf.image.decode_png(bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.expand_dims(image, 0)
        my_model_pred = my_model(image)[0]
        another_model_pred = another_model(image)[0]
        print(my_model_pred.numpy())
        print(another_model_pred.numpy())
        print('*' * 30)



if __name__ == '__main__':
    main()
