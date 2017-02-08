# -*- coding: utf-8 -*-
from __future__ import absolute_import

import keras.backend as K
from keras import activations, initializations, regularizers
from keras.layers import GRU, LSTM, SimpleRNN


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


def recurrent(output_dim, model='lstm', activation='tanh', regularizer=None,
              dropout=0., zoneout=0., **kwargs):
    if model == 'rnn':
        return SimpleRNN(output_dim, activation=activation,
                         W_regularizer=regularizer, U_regularizer=regularizer,
                         dropout_W=dropout, dropout_U=dropout,
                         zoneout_h=zoneout, consume_less='gpu', **kwargs)
    if model == 'gru':
        return GRU(output_dim, activation=activation,
                   W_regularizer=regularizer, U_regularizer=regularizer,
                   dropout_W=dropout, dropout_U=dropout, zoneout_h=dropout,
                   consume_less='gpu', **kwargs)
    if model == 'lstm':
        return LSTM(output_dim, activation=activation,
                    W_regularizer=regularizer, U_regularizer=regularizer,
                    dropout_W=dropout, dropout_U=dropout, zoneout_h=zoneout,
                    zoneout_c=zoneout, consume_less='gpu', **kwargs)
    if model == 'rhn':
        return RHN(output_dim, depth=1,
                     bias_init=highway_bias_initializer,
                     activation=activation, layer_norm=False, ln_gain_init='one',
                     ln_bias_init='zero', mi=False,
                     W_regularizer=regularizer, U_regularizer=regularizer,
                     dropout_W=dropout, dropout_U=dropout, consume_less='gpu' **kwargs)
    raise ValueError('model %s was not recognized' % model)
