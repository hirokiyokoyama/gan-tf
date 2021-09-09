import tensorflow as tf

def _maybe_convert_labels(y_true):
    is_binary = tf.reduce_all(tf.logical_or(
        y_true == 0, y_true == 1))
    if is_binary:
        return 2. * y_true - 1.
    else:
        return y_true

class Wasserstein(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = _maybe_convert_labels(y_true)

        return tf.reduce_mean(-y_true * y_pred)

class RandomHinge(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = _maybe_convert_labels(y_true)

        r = tf.random.uniform(tf.shape(y_pred), 0.8, 1.0)
        return tf.reduce_mean(tf.maximum(r - y_true * y_pred, 0.))
