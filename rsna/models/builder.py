import tensorflow as tf

from models.backbones import build_backbone as build_backbone_fn, KERAS_MODELS
from models.callbacks import build_callbacks as build_callbacks_fn
from models.heads import build_head as build_head_fn
from models.losses import build_loss as build_loss_fn
from models.metrics import build_metrics as build_metrics_fn
from models.optimizers import build_optimizer as build_optimizer_fn
from models.schedulers import build_scheduler as build_scheduler_fn


def build_backbone(cfg):
    return build_backbone_fn(cfg)


def build_callbacks(cfg):
    return build_callbacks_fn(cfg)


def build_head(cfg):
    return build_head_fn(cfg)


def build_loss(cfg):
    return build_loss_fn(cfg)


def build_metrics(cfg):
    return build_metrics_fn(cfg)


def build_optimizer(cfg, scheduler=None):
    if scheduler is not None:
        cfg['learning_rate'] = scheduler
    return build_optimizer_fn(cfg)


def build_scheduler(cfg):
    return build_scheduler_fn(cfg)


def build_classifier(cfg):
    backbone_cfg = cfg['backbone']

    input_tensor = tf.keras.layers.Input(shape=backbone_cfg['input_shape'])
    if backbone_cfg['type'].lower() in KERAS_MODELS:
        backbone_cfg['input_tensor'] = input_tensor
        backbone_cfg['include_top'] = False

    freeze = backbone_cfg.pop('freeze', None)
    backbone = build_backbone(backbone_cfg)
    if freeze:
        backbone.trainable = False

    head_cfg = cfg['head']
    head = build_head(head_cfg)

    inputs = backbone.inputs
    x = backbone.output
    outputs = head(x)

    classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

    #######
    # import keras_efficientnet_v2
    #
    # def normalize(image):
    #     image = tf.repeat(image, repeats=3, axis=3)
    #     image = tf.cast(image, tf.float32)
    #     image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
    #
    #     return image
    #
    # image = tf.keras.layers.Input((1024, 1024, 1), name='image', dtype=tf.uint8)
    #
    # # Normalize Input
    # image_norm = normalize(image)
    #
    # # CNN Prediction
    # outputs = keras_efficientnet_v2.EfficientNetV2T(
    #     input_shape=[1024, 1024, 3],
    #     pretrained='imagenet',
    #     num_classes=1,
    #     classifier_activation='sigmoid',
    #     dropout=0.30,
    # )(image_norm)
    #
    # classifier = tf.keras.models.Model(inputs=image, outputs=outputs)
    # classifier.load_weights('/app/test/model.h5')
    # classifier.trainable = False

    ######
    from keras_cv_attention_models import efficientnet

    base = getattr(efficientnet, 'EfficientNetV1B0')(input_shape=(1024, 1024, 3),
                                                     pretrained='imagenet',
                                                     num_classes=0)  # get base model (efficientnet), use imgnet weights
    inp = base.inputs
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # use GAP to get pooling result form conv outputs
    x = tf.keras.layers.Dense(32, activation='silu')(x)  # use activation to apply non-linearity
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # use sigmoid to convert predictions to [0-1]
    classifier = tf.keras.Model(inputs=inp, outputs=x)

    return classifier
