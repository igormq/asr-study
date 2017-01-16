import keras
import keras.backend as K

import numpy as np
import tensorflow as tf

def decode(inputs, **kwargs):
    import tensorflow as tf
    is_greedy = kwargs.get('is_greedy', True)
    y_pred, seq_len = inputs

    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

    if is_greedy:
        decoded = tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    else:
        decoded = tf.nn.ctc_beam_search_decoder(y_pred, seq_len)[0][0]

    return decoded

def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return (y_pred_shape[:1], None)

def ctc_lambda_func(args):
    y_pred, labels, input_length = args
    import tensorflow as tf
    return tf.nn.ctc_loss(tf.transpose(y_pred, perm=[1, 0, 2]), labels, input_length[:, 0])

def ctc_dummy_loss(y_true, y_pred):
    return y_pred

def decoder_dummy_loss(y_true, y_pred):
    return K.zeros((1,))
