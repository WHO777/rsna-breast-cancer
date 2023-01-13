import contextlib
from pathlib import Path

from absl import flags
from absl import logging
from absl import app

import tensorflow as tf

try:
    import wandb
    is_wandb_available = True
except ImportError:
    is_wandb_available = False

from rsna import config
from rsna.models import (build_classifier, build_metrics, build_callbacks, build_loss,
                    build_optimizer, build_scheduler)
from rsna.datasets import build_bataset
from rsna.utils.train import set_precision_policy, get_strategy
from rsna.utils.model import get_layer_from_model

FLAGS = flags.FLAGS
DEFAULT_PARENT_MODEL_DIR = str(Path('runs') / 'train')


def define_flags():
    flags.DEFINE_string('hparams', '', '')
    flags.DEFINE_string('model_dir', None, '')


def _init_devices():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _get_model_dir(name=None):
    name = name or 'train0'
    parent_model_dir = Path(DEFAULT_PARENT_MODEL_DIR)
    default_model_dir = parent_model_dir / name
    if default_model_dir.is_dir():
        default_dirs = [x for x in parent_model_dir.iterdir() if 'train' in x.stem]
        default_dir_idxs = [path.stem[5:] for path in default_dirs]
        default_dir_idxs_int = []
        for idx in default_dir_idxs:
            with contextlib.suppress(ValueError):
                idx_int = int(idx)
                default_dir_idxs_int.append(idx_int)
        if default_dir_idxs_int:
            default_dir_idxs = sorted(default_dir_idxs_int)
            name = name[:5] + str(default_dir_idxs[-1] + 1)
    return str(parent_model_dir / name)


def main(_):
    cfg = config.get_default_config()
    cfg.override(FLAGS.hparams)

    _init_devices()

    train_dataset = build_bataset(cfg['train']['dataset'].as_dict())
    # for x, y in train_dataset:
    #     print(x.dtype, y.dtype)
    # import cv2
    # for i, (x, y) in enumerate(train_dataset):
    #     for j, (xx, yy) in enumerate(zip(x, y)):
    #         if yy == 1:
    #             cv2.imwrite('/app/test/test_images/' + str(i) + '_' + str(j) + '.png', xx.numpy())

    if cfg.get('val'):
        val_dataset = build_bataset(cfg['val']['dataset'].as_dict())
    else:
        val_dataset = None

    if cfg.get('scheduler', None):
        scheduler = build_scheduler(cfg['scheduler'].as_dict())
    else:
        scheduler = None

    mixed_precision = cfg.mixed_precision
    set_precision_policy(mixed_precision)

    strategy = get_strategy(cfg.get('devices', None))
    with strategy.scope():
        model = build_classifier(cfg['model'].as_dict())
        optimizer = build_optimizer(cfg['optimizer'].as_dict(), scheduler=scheduler)
        loss = build_loss(cfg['loss'].as_dict())
        metrics = build_metrics(cfg['metrics'])
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    model_dir = FLAGS.model_dir or _get_model_dir()
    model_dir = str(Path(model_dir).absolute())

    Path(model_dir).mkdir(parents=True)
    logging.info(f'Results will be saved in {model_dir}')

    cfg_save_path = str(Path(model_dir).absolute() / 'config.yaml')
    cfg.save_to_yaml(cfg_save_path)
    logging.info(f'Config saved as {cfg_save_path}')

    for callback in cfg.get('callbacks', []):
        if isinstance(callback, (tuple, list)):
            name, params = callback
            if name == 'TensorBoard':
                logdir = params.get('log_dir', None) or 'logs'
                params['log_dir'] = str(Path(model_dir) / logdir)
            elif name == 'ModelCheckpoint':
                filepath = params.get('filepath', None) or 'checkpoints'
                params['filepath'] = str(Path(model_dir) / filepath)
            elif name == 'CSVLogger':
                filename = params.get('filename', None) or 'history.csv'
                params['filename'] = str(Path(model_dir) / filename)
        else:
            callback_type = callback['type']
            if callback_type == 'GradCam':
                num_batches = callback.get('num_batches', 1)
                last_conv_layer = callback.get('last_conv_layer', 'top_conv')
                # check if layer exist
                assert get_layer_from_model(model, last_conv_layer, recursive=True) is not None
                gradcam_ds = val_dataset if val_dataset is not None else train_dataset
                if isinstance(gradcam_ds, tf.keras.utils.Sequence):
                    dataset = tf.data.Dataset.from_tensor_slices(
                        [gradcam_ds.__getitem__(i) for i in range(num_batches)])
                else:
                    dataset = gradcam_ds.take(num_batches)
                callback['last_conv_layer'] = last_conv_layer
                callback['dataset'] = dataset
                callback['model_dir'] = model_dir

    callbacks = build_callbacks(cfg.get('callbacks', []))

    if cfg.get('wandb', None):
        assert is_wandb_available, "Please install wandb, you can do this by run 'pip3 install wandb'"
        wandb.init(project='rsna_breast_cancer', name=cfg.wandb.run_name or 'default_run')
        callbacks.append(wandb.keras.WandbMetricsLogger(log_freq='batch'))

    model.fit(train_dataset,
              epochs=cfg.train.epochs,
              callbacks=callbacks,
              validation_data=val_dataset)

    # image_paths = list(Path('/app/test/test_images').absolute().iterdir())
    # for image_path in image_paths:
    #     bytes = tf.io.read_file(str(image_path))
    #     image = tf.image.decode_png(bytes, channels=1)
    #     image = tf.image.resize(image, (1024, 1024))
    #     image = tf.expand_dims(image, 0)
    #     preds = model(image)[0]
    #     print(image_path.name, tf.argmax(preds), preds[tf.argmax(preds)].numpy())

    strategy._extended._collective_ops._pool.close()


if __name__ == '__main__':
    define_flags()
    app.run(main)
