import tensorflow as tf

def ler(y_true, y_pred, **kwargs):
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))
