# -*- coding: utf-8 -*-
from __future__ import absolute_import

import keras.backend as K
import tensorflow as tf

from keras import activations, initializations, regularizers
from keras.layers import GRU, SimpleRNN
from keras.layers import LSTM as keras_LSTM


def highway_bias_initializer(shape, name=None):
    return -2 * initializations.one(shape, name=name)


def layer_normalization(x, gain, bias, epsilon=1e-5):
    m = K.mean(x, axis=-1, keepdims=True)
    std = K.std(x, axis=-1, keepdims=True)
    x_normed = (x - m) / (std + epsilon) * gain + bias
    return x_normed


def multiplicative_integration_init(shape, alpha_init='one',
                                    beta1_init='one', beta2_init='one',
                                    name='mi', has_input=True):

    beta1 = initializations.get(beta1_init)(shape, name='%s_beta1' % name)
    if has_input:
        alpha = initializations.get(alpha_init)(shape, name='%s_alpha' % name)
        beta2 = initializations.get(beta2_init)(shape, name='%s_beta2' % name)
        return alpha, beta1, beta2

    return beta1


def multiplicative_integration(Wx, Uz, params, has_input=True):
    if has_input:
        alpha, beta1, beta2 = params
        return alpha * Wx * Uz + beta1 * Uz + beta2 * Wx

    beta1 = params
    return beta1 * Uz

def to_dense(x):
    if K.is_sparse(x):
        return tf.sparse_tensor_to_dense(x, default_value=-1)
    return x


def to_dense_output_shape(input_shape):
    return input_shape


LN = layer_normalization
mi = multiplicative_integration
mi_init = multiplicative_integration_init
