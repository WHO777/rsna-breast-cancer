import tensorflow as tf

from .builder import METRICS


@METRICS.register_module()
class pFBeta(tf.keras.metrics.Metric):
    """Compute overall probabilistic F-beta score."""

    def __init__(self, beta=1, epsilon=1e-5, name='pF1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
        self.pos = self.add_weight(name='pos', initializer='zeros')
        self.ctp = self.add_weight(name='ctp', initializer='zeros')
        self.cfp = self.add_weight(name='cfp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        pos = tf.reduce_sum(y_true)
        ctp = tf.reduce_sum(y_pred[y_true == 1])
        cfp = tf.reduce_sum(y_pred[y_true == 0])
        self.pos.assign_add(pos)
        self.ctp.assign_add(ctp)
        self.cfp.assign_add(cfp)

    def result(self):
        beta_squared = self.beta * self.beta
        c_precision = self.ctp / (self.ctp + self.cfp + self.epsilon)
        c_recall = self.ctp / (self.pos + self.epsilon)
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return tf.cond(c_precision > 0 and c_recall > 0, lambda: result, lambda: 0.0)
