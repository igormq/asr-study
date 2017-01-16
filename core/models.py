from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core.ctc_utils as ctc_utils
from common.hparams import HParams

import keras
from keras.initializations import uniform
from keras.models import Model
from keras.layers import Input, GaussianNoise, TimeDistributed, Dense, LSTM, Masking, Bidirectional, Lambda

def ctc_model(input_, output):
    ''' Given the output returns a model appending ctc_loss and the decoder
    '''

    # Define placeholders
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    input_length = Input(name='input_length', shape=(None,), dtype='int32')

    # Define a decoder
    dec = Lambda(ctc_utils.decode, output_shape=ctc_utils.decode_output_shape,
                 arguments={'is_greedy': True}, name='decoder')
    y_pred = dec([output, input_length])

    ctc = Lambda(ctc_utils.ctc_lambda_func, output_shape=(1,), name="ctc")
    # Define loss as a layer
    l = ctc([output, labels, input_length])

    return Model(input=[input_, labels, input_length], output=[l, y_pred])

def graves2006(hparams=None):
    params = HParams(nb_features=26,
                     nb_hidden=100,
                     nb_classes=28,
                     std=.6,
                     lr=0.01)

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x
    # o = Masking(mask_value=0.)(o)
    o = GaussianNoise(params.std)(o)
    o = Bidirectional(LSTM(params.nb_hidden,
                      return_sequences=True,
                      consume_less='gpu'))(o)
    o = TimeDistributed(Dense(params.nb_classes))(o)

    return ctc_model(x, o)
