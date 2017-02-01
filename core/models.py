from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core.ctc_utils as ctc_utils
from common.hparams import HParams

import keras
from keras.initializations import uniform
from keras.models import Model
from keras.layers import Input, GaussianNoise, TimeDistributed, Dense, LSTM, Masking, Bidirectional, Lambda, Dropout
from keras.regularizers import l1, l2, l1l2

def ctc_model(input_, output, **kwargs):
    """ Given the input and output returns a model appending ctc_loss and the decoder

    # Arguments
        see core.ctc_utils.layer_utils.decode for more arguments
    """

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
    """ Implementation of Graves' model
    Reference:
        [1] Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." Proceedings of the 23rd international conference on Machine learning. ACM, 2006.
    """
    params = HParams(nb_features=26,
                     nb_hidden=100,
                     nb_classes=28,
                     std=.6)

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

def bayesian_lstm(hparams):
    """ LSTM with variational dropout and weight decay. Following the best topology of [2] (without a transducer).
    Note:
        Dropout is tied through layers, and weights and the same for weight decay, minimizing the number of hyper parameters
    Reference:
        [1] Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
        [2] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech recognition with deep recurrent neural networks", 2013.
    """
    params = HParams(nb_features=39,
                     nb_classes=28,
                     nb_hidden=256,
                     nb_layers=3,
                     dropout=0.25,
                     input_dropout=False,
                     weight_decay=1e-4)

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    if params.input_dropout:
        o = Dropout(params.dropout)(o)

    for _ in range(params.nb_layers):
        o = Bidirectional(LSTM(params.nb_hidden,
                          return_sequences=True,
                          W_regularizer=l2(params.weight_decay),
                          U_regularizer=l2(params.weight_decay),
                          dropout_W=params.dropout,
                          dropout_U=params.dropout,
                          consume_less='gpu'))(o)

    o = TimeDistributed(Dense(params.nb_classes,
                              W_regularizer=l2(params.weight_decay)))(o)

    return ctc_model(x, o)
