from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core.ctc_utils as ctc_utils
from common.hparams import HParams

import keras
import keras.backend as K
from keras.initializations import uniform
from keras.activations import relu

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

def eyben(hparams=None):
    """ Implementation of Eybens' model
    Reference:
        [1] Eyben, Florian, et al. "From speech to letters-using a novel neural network architecture for grapheme based asr." Automatic Speech Recognition & Understanding, 2009. ASRU 2009. IEEE Workshop on. IEEE, 2009.
    """
    params = HParams(nb_features=39,
                     nb_hidden=[78, 120, 27],
                     nb_classes=28)

    params.parse(hparams)

    assert len(params.nb_hidden) == 3

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    if params.nb_hidden[0]:
        o = TimeDistributed(Dense(params.nb_hidden[0]))(o)
    if params.nb_hidden[1]:
        o = Bidirectional(LSTM(params.nb_hidden[1],
                          return_sequences=True,
                          consume_less='gpu'))(o)
    if params.nb_hidden[2]:
        o = Bidirectional(LSTM(params.nb_hidden[2],
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


def deep_speech(hparams):
    """ Deep Speech model.

        Contains five layers: 3 FC - BRNN - 1 FC
        Dropout only applied to fully connected layers (between 5% to 10%)

    Note:
        * We are not translating the raw audio files by 5 ms (Sec 2.1 in [1])
        * We are not striding the RNN to halve the timesteps (Sec 3.3 in [1])
        * We are not using frames of context
        * Their output contains {a, ..., z, space, apostrophe, blank}
    Experiment 5.1: Conversational speech: Switchboard Hub5'00 (full)
        * Input - 80 linearly spaced log filter banks and an energy term. The
        filter banks are computed over windows of 20ms strided by 10ms.
        * Speaker adaptation - spectral features are normalized on a per
        speaker basis.
        * Hidden units: {2304, 2048}
        * Essemble of 4 networks
    Experiment 5.2: Noisy speech
        * Input - 160 linearly spaced log filter banks. The filter banks are
        computed over windows of 20ms strided by 10ms. Global mean and standard
        deviation over training set normalization
        * Speaker adaptation - none
        * Hidden units: 2560
        * Essemble of 6 networks
    Reference:
        [1] HANNUN, A. Y. et al. Deep Speech: Scaling up end-to-end speech
        recognition. arXiV, 2014.
    """
    params = HParams(nb_features=81,
                     nb_classes=29,
                     nb_hidden=2048,
                     dropout=0.1,
                     max_value=20)

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    def clipped_relu(x):
        return relu(x, max_value=params.max_value)

    # First layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(params.dropout))(o)

    # Second layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(params.dropout))(o)

    # Third layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(params.dropout))(o)

    # Fourth layer
    o = Bidirectional(SimpleRNN(params.nb_hidden, return_sequences=True,
                                dropout_W=params.dropout,
                                activation=clipped_relu,
                                init='he_normal'), merge_mode='sum')(o)
    o = TimeDistributed(Dropout(params.dropout))(o)

    # Fifth layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(params.dropout))(o)

    # Output layer
    o = TimeDistributed(Dense(params.nb_classes))(o)

    return ctc_model(x, o)


def maas(hparams):
    """ Maas' model.
    Reference:
        [1] Maas, Andrew L., et al. "Lexicon-Free Conversational Speech Recognition with Neural Networks." HLT-NAACL. 2015.
    """
    params = HParams(nb_features=81,
                     nb_classes=29,
                     nb_hidden=1824,
                     dropout=0.1,
                     max_value=20)

    params.parse(hparams)

    x = Input(name='input', shape=(None, params.nb_features))
    o = x

    def clipped_relu(x):
        return relu(x, max_value=params.max_value)

    # First layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)

    # Second layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)

    # Third layer
    o = Bidirectional(SimpleRNN(params.nb_hidden, return_sequences=True,
                                dropout_W=params.dropout,
                                activation=clipped_relu,
                                init='he_normal'), merge_mode='sum')(o)

    # Fourth layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)

    # Fifth layer
    o = TimeDistributed(Dense(params.nb_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)

    # Output layer
    o = TimeDistributed(Dense(params.nb_classes))(o)

    return ctc_model(x, o)
