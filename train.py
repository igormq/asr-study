from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import codecs

from sklearn.model_selection import train_test_split

import keras
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from core import metrics
from core.ctc_utils import ctc_dummy_loss, decoder_dummy_loss
from core.callbacks import MetaCheckpoint

from preprocessing import audio, text

from common.utils import get_functions_from_module
from common.dataset_generator import DatasetGenerator
from common.hparams import HParams

import argparse
import uuid
import os
import json
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an ASR system.')

    parser.add_argument('--model', default='graves2006', type=str)
    parser.add_argument('--params', default='{}', type=str)

    parser.add_argument('--nb_epoch', default=100, type=int)

    parser.add_argument('--dataset', default=None, type=str, help="Path to dataset h5 or json file separated by train/valid/test")

    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--valid', type=str, default=None)

    parser.add_argument('--split', type=float, default=.2, help='Split valid/train ratio. Only enabled when valid=None')

    parser.add_argument('--lr', default=0.01, type=float)

    parser.add_argument('--lr_schedule', default='ReduceLROnPlateau')
    parser.add_argument('--lr_params', default="{'monitor': 'val_loss', 'factor':0.1, 'patience':5, 'min_lr':1e-6}")

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clipnorm', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'])

    parser.add_argument('--feats', type=str, default='raw', choices=['mfcc', 'raw', 'logfbank'])
    parser.add_argument('--feats_params', type=str, default='{}')

    parser.add_argument('--text_parser', type=str, default='simple_char_parser')

    parser.add_argument('--save', default=None, type=str)

    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    # Choosing gpu
    if args.gpu == '-1':
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        if args.gpu == 'all':
            args.gpu = ''
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = args.gpu
    if args.allow_growth == True:
        config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    # Recovering all valid models
    valid_models = get_functions_from_module('core.models')
    if args.model not in valid_models.keys():
        raise ValueError('model %s not found. Valid models are: %s' % (args.model, ', '.join(valid_models.keys())))

    # Load model
    model = valid_models[args.model](args.params)

    # Optimization
    if args.opt == 'sgd':
        opt = SGD(lr=args.lr, momentum=args.momentum, clipnorm=args.clipnorm)
    elif args.opt == 'adam':
        opt = Adam(lr=args.lr, clipnorm=args.clipnorm)

    # Compile with dummy loss
    model.compile(loss={'ctc': ctc_dummy_loss,
                        'decoder': decoder_dummy_loss},
                  optimizer=opt, metrics={'decoder': metrics.ler},
                  loss_weights=[1, 0])

    # Creating the results folder
    name = args.save
    if name is None:
        name = os.path.join('results', '%s_%s' % (args.model, datetime.datetime.now()))
    if not os.path.isdir(name):
        os.makedirs(name)

    # Callbacks
    meta_ckpt = MetaCheckpoint(os.path.join(name, 'meta.yaml'), training_args=vars(args))
    model_ckpt = ModelCheckpoint(os.path.join(name, 'model.h5'))
    best_ckpt = ModelCheckpoint(os.path.join(name, 'best.h5'), monitor='val_decoder_ler', save_best_only=True, mode='min')
    callback_list = [meta_ckpt, model_ckpt, best_ckpt]

    # LR schedules
    lr_params = HParams()
    lr_params.parse(args.lr_params)
    if args.lr_schedule is not None and args.lr_schedule == 'ReduceLROnPlateau':
        reduce_lr = ReduceLROnPlateau(*lr_params.keyvals)
        callback_list.append(reduce_lr)

    # Features extractor
    feats_params = HParams()
    feats_params.parse(args.feats_params)
    if args.feats == 'mfcc':
        feats_extractor = audio.MFCC(*feats_params.keyvals)
    elif args.feats == 'logfbank':
        feats_extractor = audio.LogFbank(*feats_params.keyvals)
    else:
        feats_extractor = None

    # Recovering text parser
    valid_text_parsers = get_functions_from_module('preprocessing.text')
    if not valid_text_parsers.has_key(args.text_parser):
        raise ValueError('text_parser %s not found. Valid text_parser are: %s' % (args.text_parser, ', '.join(valid_text_parsers.keys())))

    text_parser = valid_text_parsers[args.text_parser]

    # Data generator
    data_gen = DatasetGenerator(feats_extractor, text_parser)

    train_flow, valid_flow, test_flow = None, None, None

    if args.dataset is not None:
        train_flow, valid_flow, test_flow = data_gen.flows_from_fname(args.dataset, batch_size=args.batch_size, seed=0)
    else:
        if args.train:
            train_flow = data_gen.flow_from_fname(args.train, dt_name='train', batch_size=args.batch_size, seed=0)

        if args.valid:
            valid_flow = data_gen.flow_from_fname(args.valid, dt_name='valid', batch_size=args.batch_size, seed=0)

        if args.test:
             test_flow = data_gen.flow_from_fname(args.test, dt_name='test', batch_size=args.batch_size)

    nb_val_samples = None
    if valid_flow:
        nb_val_samples = valid_flow.len

    # Fit the model
    model.fit_generator(train_flow, samples_per_epoch=train_flow.len, nb_epoch=args.nb_epoch, validation_data=valid_flow, nb_val_samples=nb_val_samples, max_q_size=10, nb_worker=1, callbacks=callback_list, verbose=1)

    if test_flow:
        metrics = model.evaluate_generator(test_flow, test_flow.len, max_q_size=10, nb_worker=1)
