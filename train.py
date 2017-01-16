from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import codecs

from sklearn.model_selection import train_test_split

import keras
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from core import metrics
from core.ctc_utils import ctc_dummy_loss, decoder_dummy_loss
from core.callbacks import MetaCheckpoint

from preprocessing import audio, text

from common.utils import get_functions_from_module
from common.dataset_generator import DatasetGenerator

import argparse
import uuid
import os
import json


def from_json(path):

    with codecs.open(path, 'r', encoding='utf8') as f:
        data = json.load(f)

    audio_paths = [d[0] for d in data]
    texts = [d[1] for d in data]

    return audio_paths, texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an ASR system.')

    parser.add_argument('--model', default='graves2006', type=str)
    parser.add_argument('--params', default='{}', type=str)

    parser.add_argument('--nb_epoch', default=100, type=int)

    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--valid', type=str, default=None)

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clipnorm', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--save', default=os.path.join('results', str(uuid.uuid1())), type=str)

    parser.add_argument('--gpu', default='0', type=str)

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

    name = args.save
    if not os.path.isdir(name):
        os.makedirs(name)

    valid_models = get_functions_from_module('core.models')
    if args.model not in valid_models.keys():
        raise ValueError('model %s not found. Valid models are: %s' % (args.model, ', '.join(valid_models.keys())))

    model = valid_models[args.model](args.params)

    # Optimization
    opt = SGD(lr=args.lr, momentum=args.momentum, clipnorm=args.clipnorm)

    # Compile with dummy loss
    model.compile(loss={'ctc': ctc_dummy_loss,
                        'decoder': decoder_dummy_loss},
                  optimizer=opt, metrics={'decoder': metrics.ler},
                  loss_weights=[1, 0])

    # Callbacks
    meta_ckpt = MetaCheckpoint(os.path.join(name, 'meta.yaml'), training_args=vars(args))
    model_ckpt = ModelCheckpoint(os.path.join(name, 'model.h5'))
    best_ckpt = ModelCheckpoint(os.path.join(name, 'best.h5'), monitor='val_decoder_ler', save_best_only=True, mode='min')
    callback_list = [meta_ckpt, model_ckpt, best_ckpt]

    # Data generator
    data_gen = DatasetGenerator(lambda x: audio.MFCC(d=True)(x), text.simple_parser)

    X, y = from_json(args.train)

    if args.valid is None:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)
    else:
        X_train, y_train = X, y
        X_valid, y_valid = from_json(args.valid)

    # #  Fit the model
    model.fit_generator(data_gen.flow(X_train, y_train, batch_size=args.batch_size, seed=0), samples_per_epoch=len(X_train), nb_epoch=args.nb_epoch, validation_data=data_gen.flow(X_valid, y_valid, batch_size=args.batch_size, seed=0), nb_val_samples=len(X_valid), max_q_size=10, nb_worker=1, callbacks=callback_list, verbose=1)
