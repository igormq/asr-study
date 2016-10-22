import os

import tensorflow as tf
# Import CTC loss
# try:
#     from tensorflow.contrib.warpctc.ctc_ops import warp_ctc_loss as ctc_loss
#     print('Using warp ctc :)')
# except ImportError:
from tensorflow.python.ops.ctc_ops import ctc_loss
print('Using tf ctc :(')

import numpy as np
import librosa
from scipy import sparse
import argparse
import h5py
import uuid

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda, TimeDistributed, LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from layers import RHN

from preprocess_timit import get_char_map, get_phn_map

def get_inv_dict(dict_label):
    inv_dict = {v: k for (k, v) in dict_label.iteritems()}
    # Add blank label
    inv_dict[len(inv_dict)] = '<b>'
    return inv_dict

def get_output(model, x):
    return ''.join([inv_dict[i] for i in model.predict(x).argmax(axis=2)[0]])

def get_from_h5(h5_file, dataset, label_type='phn'):
    X = np.array(h5_file['%s/inputs/data' %dataset])
    seq_len = np.array(h5_file['%s/inputs/seq_len' %dataset])

    values = np.array(h5_file['%s/%s/values' %(dataset, label_type)])
    indices = np.array(h5_file['%s/%s/indices' %(dataset, label_type)])
    indices = (indices[:, 0], indices[:, 1])
    shape = np.array(h5_file['%s/%s/shape' %(dataset, label_type)])

    y = sparse.coo_matrix((values, indices), shape=shape).tolil()
    return X, seq_len, y

def ctc_lambda_func(args):
    y_pred, labels, input_length = args
    return ctc_loss(tf.transpose(y_pred, perm=[1, 0, 2]), labels, input_length[:, 0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CTC Models training with TIMIT dataset')
    parser.add_argument('--layer', type=str, choices=['lstm', 'rhn', 'rnn',
                                                      'gru'], default='lstm')
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--layer_norm', action='store_true', default=False)
    parser.add_argument('--nb_hidden', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nb_epoch', type=int, default=250)
    parser.add_argument('--label_type', type=str, choices=['phn', 'char'], default='phn')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clipnorm', type=float, default=10.)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = args.gpu
    session = tf.Session(config=config)
    K.set_session(session)


    if args.layer == 'lstm':
        RNN = LSTM
    elif args.layer == 'gru':
        RNN = GRU
    elif args.layer == 'rnn':
        RNN = SimpleRNN
    elif args.layer == 'rhn':
        RNN = RHN

    # Read dataset
    with h5py.File('timit.h5', 'r') as f:
        X, seq_len, y = get_from_h5(f, 'train', label_type=args.label_type)
        X_valid, seq_len_valid, y_valid = get_from_h5(f, 'valid', label_type=args.label_type)
        X_test, seq_len_test, y_test = get_from_h5(f, 'test', label_type=args.label_type)

    if args.label_type == 'phn':
        _, dict_label = get_phn_map('timit/phones.60-48-39.map')
    else:
        dict_label = get_char_map()

    inv_dict = get_inv_dict(dict_label)

    nb_features = X.shape[2]
    nb_classes = len(inv_dict)

    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")

    # Define placeholders
    x = Input(name='input', shape=(None, nb_features))
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    input_length = Input(name='input_length', shape=(None,), dtype='int32')

    # Define model
    o = x
    if args.layer == 'rhn':
        o = RHN(args.nb_hidden, nb_layers=args.nb_layers,
                return_sequences=True, layer_norm=args.layer_norm)(o)
    else:
        for l in xrange(args.nb_layers):
            o = RNN(args.nb_hidden, return_sequences=True)(o)

    o = TimeDistributed(Dense(nb_classes))(o)
    # Define loss as a layer
    l = ctc([o, labels, input_length])

    model = Model(input=[x, labels, input_length], output=l)
    pred = Model(input=x, output=o)

    # Optimization
    opt = SGD(lr=args.lr, momentum=args.momentum, clipnorm=args.clipnorm)
    # Compile with dummy loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

    #  Fit the model
    history = model.fit([X, y, seq_len], np.zeros((X.shape[0],)),
                        batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                        validation_data=([X_valid, y_valid, seq_len_valid],
                                         np.zeros((X_valid.shape[0],))))

    meta = {'history': history.history, 'params': vars(args)}

    if not os.path.isdir('results'):
        os.makedirs('results')

    name = os.path.join('results', str(uuid.uuid1()))
    print('Saving at ./%s' % name)
    with open(name, 'wb') as f:
        pickle.dump(meta, f)

    model.save('%s.h5' % name)
