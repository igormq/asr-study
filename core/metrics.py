import tensorflow as tf

def ler(y_true, y_pred, **kwargs):
    """
        Label Error Rate. For more information see 'tf.edit_distance'
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))
