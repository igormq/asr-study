from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core.ctc_utils as ctc_utils
from common.hparams import HParams

import keras
import keras.backend as K
from keras.initializations import uniform

from keras.models import Model

from keras.layers import Input
from keras.layers import GaussianNoise
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import merge

from keras.regularizers import l1, l2, l1l2

from .layers import recurrent


def ctc_model(input_, output, **kwargs):
    """ Given the input and output returns a model appending ctc_loss and the
    decoder

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
    loss = ctc([output, labels, input_length])

    return Model(input=[input_, labels, input_length], output=[loss, y_pred])


def graves2006(hparams=None):
    """ Implementation of Graves' model
    Reference:
        [1] Graves, Alex, et al. "Connectionist temporal classification:
        labelling unsegmented sequence data with recurrent neural networks."
        Proceedings of the 23rd international conference on Machine learning.
        ACM, 2006.
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
    """ LSTM with variational dropout and weight decay. Following the best
    topology of [2] (without a transducer).
    Note:
        Dropout is tied through layers, and weights and the same for weight
        decay, minimizing the number of hyper parameters
    Reference:
        [1] Gal, Y, "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks", 2015.
        [2] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech
        recognition with deep recurrent neural networks", 2013.
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


def zoneout_rnn(hparams):
    """ LSTM/GRU with variational dropout, weight decay and zoneout. Following the
    best topology of [2] (without a transducer).
    Note:
        Dropout, zoneout and weight decay is tied through layers, minimizing
        the number of hyper parameters
    Reference:
        [1] Gal, Y, "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks", 2015.
        [2] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech
        recognition with deep recurrent neural networks", 2013.
        [3] Krueger, David, et al. "Zoneout: Regularizing rnns by randomly
        preserving hidden activations", 2016.
    """
    params = HParams(nb_features=39,
                     nb_classes=28,
                     nb_hidden=256,
                     nb_layers=3,
                     dropout=0.,
                     zoneout=0.2,
                     input_dropout=False,
                     input_std_noise=.0,
                     weight_decay=1e-4,
                     model='lstm')

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    if params.input_std_noise is not None:
        o = GaussianNoise(params.input_std_noise)(o)

    if params.input_dropout:
        o = Dropout(params.dropout)(o)

    for _ in range(params.nb_layers):
        o = Bidirectional(recurrent(params.nb_hidden,
                                    model=params.model,
                                    return_sequences=True,
                                    regularizer=l2(params.weight_decay),
                                    dropout=params.dropout,
                                    zoneout=params.zoneout))(o)

    o = TimeDistributed(Dense(params.nb_classes,
                              W_regularizer=l2(params.weight_decay)))(o)

    return ctc_model(x, o)

def imlstm(hparams):
    """ Improved LSTM:
        * Residual connection
        * Variational Dropout
        * Zoneout
        * Layer Normalization
        * Multiplicative Integration
    Note:
        Dropout, zoneout and weight decay is tied through layers, in order to
        minimizing the number of hyper parameters
    Reference:
        [1] Gal, Y, "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks", 2015.
        [2] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech
        recognition with deep recurrent neural networks", 2013.
        [3] Krueger, David, et al. "Zoneout: Regularizing rnns by randomly
        preserving hidden activations", 2016.
        [4] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer
        normalization.", 2016.
        [5] Wu, Yuhuai, et al. "On multiplicative integration with recurrent
        neural networks." Advances In Neural Information Processing Systems.
        2016.
        [6] Wu, Yonghui, et al. "Google's Neural Machine Translation System:
        Bridging the Gap between Human and Machine Translation.", 2016.
    """
    params = HParams(nb_features=120,
                     nb_classes=28,
                     nb_hidden=320,
                     nb_layers=4,
                     dropout=0.,
                     zoneout=0.,
                     input_dropout=False,
                     input_std_noise=.0,
                     weight_decay=1e-4,
                     res_con=False,
                     layer_norm=False, ln_init=['one', 'zero'],
                     mi=False, mi_init=['one', 'one', 'one'],
                     activation='tanh')

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    if params.input_std_noise is not None:
        o = GaussianNoise(params.input_std_noise)(o)

    if params.res_con:
        o = TimeDistributed(Dense(params.nb_hidden*2,
                                  W_regularizer=l2(params.weight_decay)))(o)

    if params.input_dropout:
        o = Dropout(params.dropout)(o)

    for i, _ in enumerate(range(params.nb_layers)):
        new_o = Bidirectional(recurrent(params.nb_hidden,
                                        model='lstm',
                                        return_sequences=True,
                                        regularizer=l2(params.weight_decay),
                                        dropout=params.dropout,
                                        zoneout=params.zoneout,
                                        mi=params.mi, mi_init=params.mi_init,
                                        layer_norm=params.layer_norm,
                                        ln_init=params.ln_init,
                                        activation=params.activation))(o)

        if params.res_con:
            o = merge([new_o,  o], mode='sum')
        else:
            o = new_o

    o = TimeDistributed(Dense(params.nb_classes,
                              W_regularizer=l2(params.weight_decay)))(o)

    return ctc_model(x, o)
