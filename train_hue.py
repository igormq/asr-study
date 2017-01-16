import sys
import os
import yaml

sys.path = [os.path.join('keras')] + sys.path

import tensorflow as tf
from tensorflow.python.ops.ctc_ops import ctc_loss

import numpy as np
import librosa
import argparse
import h5py
import uuid
import cPickle as pickle

import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda, TimeDistributed, LSTM, Bidirectional
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from layers import RHN

from preprocess_hue import get_char_map
from sklearn.model_selection import train_test_split

from layers import highway_bias_initializer
keras.initializations.highway_bias_initializer = highway_bias_initializer

def get_output(model, x):
    return ''.join([inv_dict[i] for i in model.predict(x).argmax(axis=2)[0]])





def treta_loader(treta_path):
    keras.metrics.ler = ler
    keras.objectives.decoder_dummy_loss = decoder_dummy_loss
    keras.objectives.ctc_dummy_loss = ctc_dummy_loss
    from layers import RHN, highway_bias_initializer
    keras.initializations.highway_bias_initializer = highway_bias_initializer
    modelin = keras.models.load_model(treta_path, custom_objects={'RHN':RHN})
    return modelin




def ctc_model(nb_features, nb_hidden, nb_layers, nb_classes, dropout, bidirectional):
    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")
    dec = Lambda(decode, output_shape=decode_output_shape,
                 arguments={'is_greedy': True}, name='decoder')

    # Define placeholders
    x = Input(name='input', shape=(None, nb_features))
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    input_length = Input(name='input_length', shape=(None,), dtype='int32')
    o = GaussianNoise(.6)(x)

    # Define model
    o = x
    for l in xrange(args.nb_layers):
        rnn = RNN(nb_hidden, return_sequences=True, consume_less='gpu', dropout_W=dropout, dropout_U=dropout)
        o = Bidirectional(rnn)(o)

    o = TimeDistributed(Dense(nb_classes))(o)
    # Define loss as a layer
    l = ctc([o, labels, input_length])
    y_pred = dec([o, input_length])

    model = Model(input=[x, labels, input_length], output=[l, y_pred])
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CTC Models training with HUE dataset')
    parser.add_argument('--layer', type=str, choices=['lstm', 'rnn',
                                                      'gru'], default='lstm')
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--bi', action='store_true', default=False)
    parser.add_argument('--nb_hidden', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nb_epoch', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clipnorm', type=float, default=10.)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--save', default=os.path.join('results', str(uuid.uuid1())), type=str)
    parser.add_argument('--load', default=None)
    args = parser.parse_args()

    if args.gpu == '-1':
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        if args.gpu == 'all':
            args.gpu = ''
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = args.gpu

    session = tf.Session(config=config)
    K.set_session(session)

    # Read dataset
    with h5py.File('hue.h5', 'r') as f:
        X, seq_len, y = get_from_h5(f, 'train')
        X_test, seq_len_test, y_test = get_from_h5(f, 'test')

    X_train, X_valid, seq_len_train, seq_len_valid, y_train, y_valid = train_test_split(X, seq_len, y, test_size=0.15, random_state=42)

    dict_label = get_char_map()

    inv_dict = get_inv_dict(dict_label)

    nb_features = X.shape[2]
    nb_classes = len(inv_dict)

    if args.layer == 'lstm':
        RNN = LSTM
    elif args.layer == 'gru':
        RNN = GRU
    elif args.layer == 'rnn':
        RNN = SimpleRNN

    if args.load is None:
        model = ctc_model(nb_features, args.nb_hidden, args.nb_layers, nb_classes, args.dropout, args.bi)
    else:
        model = treta_loader(args.load)

    # Optimization
    opt = SGD(lr=args.lr, momentum=args.momentum, clipnorm=args.clipnorm)

    # Compile with dummy loss
    model.compile(loss={'ctc': ctc_dummy_loss,
                        'decoder': decoder_dummy_loss},
                  optimizer=opt, metrics={'decoder': ler},
                  loss_weights=[1, 0])



    name = args.save
    if not os.path.isdir(name):
        os.makedirs(name)

    # Define callbacks
    meta_ckpt = MetaCheckpoint(os.path.join(name, 'meta.yaml'), training_args=vars(args))
    model_ckpt = ModelCheckpoint(os.path.join(name, 'model.h5'))
    best_ckpt = ModelCheckpoint(os.path.join(name, 'best.h5'), monitor='val_decoder_ler', save_best_only=True, mode='min')
    callback_list = [meta_ckpt, model_ckpt, best_ckpt]

    #  Fit the model
    model.fit([X_train, y_train, seq_len_train], [np.zeros((X_train.shape[0],)), y_train],batch_size=args.batch_size, nb_epoch=args.nb_epoch, validation_data=([X_valid, y_valid, seq_len_valid], [np.zeros((X_valid.shape[0],)), y_valid]), callbacks=callback_list, shuffle=True, verbose=2)
