import tensorflow as tf

from .builder import HEADS


@HEADS.register_module()
class SimpleHead(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 num_hidden_units=None,
                 batch_norm=False,
                 relu=False,
                 dropout_rate=None,
                 name='SimpleHead'
                 ):
        super(SimpleHead, self).__init__(name=name)
        self.num_classes = num_classes
        self.num_hidden_units = num_hidden_units
        self.batch_norm = batch_norm
        self.relu = relu
        self.dropout_rate = dropout_rate

        layers = [tf.keras.layers.GlobalAveragePooling2D()]
        if num_hidden_units is not None:
            layers.append(tf.keras.layers.Dense(num_hidden_units))
        if batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        if relu:
            layers.append(tf.keras.layers.ReLU())
        if dropout_rate is not None:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
        self.model = tf.keras.Sequential(layers)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super(SimpleHead, self).get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_hidden_units": self.num_hidden_units,
            "batch_norm": self.batch_norm,
            "relu": self.relu,
            "dropout_rate": self.dropout_rate
        })
        return config
