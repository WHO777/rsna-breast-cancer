from efficientdet import hparams_config

Config = hparams_config.Config


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
    h.train.dataset.as_generator = False
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
    h.val.dataset.as_generator = False
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

