import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

from rsna.utils import image as image_utils


def get_gradcam(image, grad_model, alpha=0.4, pred_index=0):
    heatmap = gen_gradcam_heatmap(image, grad_model, pred_index=pred_index)

    heatmap = tf.cast(255 * heatmap, tf.uint8)
    heatmap = tf.cast(heatmap, tf.int32)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = tf.gather(tf.convert_to_tensor(jet_colors), heatmap)

    jet_heatmap = tf.image.resize(jet_heatmap, size=(image.shape[1], image.shape[0]))

    superimposed_img = jet_heatmap * alpha + image_utils.scale_image(image, 0, 1)

    superimposed_img = image_utils.scale_image(superimposed_img, 0, 255)
    superimposed_img = tf.cast(superimposed_img, tf.uint8)
    return superimposed_img


def gen_gradcam_heatmap(image, grad_model, pred_index=0):
    with tf.GradientTape() as tape:
        features, preds = grad_model(tf.expand_dims(image, 0))
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, features)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    features = features[0]
    heatmap = features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap
