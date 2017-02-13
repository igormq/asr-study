from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Preventing pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import warpctc_tensorflow
except ImportError:
    logging.warning('warpctc binding for tensorflow not found. :(')

import tensorflow as tf
import codecs

from sklearn.model_selection import train_test_split

import keras

import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau

from core import metrics
from core.ctc_utils import ctc_dummy_loss, decoder_dummy_loss
from core.callbacks import MetaCheckpoint, ProgbarLogger

from preprocessing import audio, text

from common.dataset_generator import DatasetGenerator
from common.hparams import HParams

import argparse
import uuid
import sys
import json
import datetime
import inspect

import logging
import common.utils as utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training an ASR system.')

    # Resume training
    parser.add_argument('--load', default=None, type=str)

    # Model settings
    parser.add_argument('--model', default='graves2006', type=str)
    parser.add_argument('--model_params', default='{}', type=str)

    # Hyper parameters
    parser.add_argument('--nb_epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clipnorm', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'])  # optimizer
    # End of hyper parameters

    # Dataset definitions
    parser.add_argument('--dataset', default=None, type=str,
                        help="Path to dataset h5 or json file \
                        separated by train/valid/test")
    # If --dataset was not defined
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--valid', type=str, default=None)

    # Features generation (if necessary)
    parser.add_argument('--feats', type=str, default='raw',
                        choices=['mfcc', 'raw', 'logfbank'])
    parser.add_argument('--feats_params', type=str, default='{}')

    # Label generation (if necessary)
    parser.add_argument('--text_parser', type=str,
                        default='simple_char_parser')
    parser.add_argument('--text_parser_params', type=str, default='{}')

    # Callbacks
    parser.add_argument('--lr_schedule', default=None)
    parser.add_argument('--lr_params',
                        default="{'monitor': 'val_loss', \
                        'factor':0.1, 'patience':5, 'min_lr':1e-6}")

    # Other configs
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')
    parser.add_argument('--verbose', default=0, type=int)

    args = parser.parse_args()

    # Setup logging
    utils.setup_logging()
    logger = logging.getLogger(__name__)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # hack in ProgbarLogger: avoid logger.infoing the dummy losses
    keras.callbacks.ProgbarLogger = lambda: ProgbarLogger(
        show_metrics=['loss', 'decoder_ler', 'val_loss', 'val_decoder_ler'])

    # GPU configuration
    utils.config_gpu(args.gpu, args.allow_growth,
                     log_device_placement=args.verbose > 1)

    # Initial configuration
    epoch_offset = 0
    meta = None

    if args.load:
        args_nondefault = utils.parse_nondefault_args(args,
                                                      parser.parse_args([]))

        logger.info('Loading model...')
        model, meta = utils.load_model(args.load, return_meta=True)

        logger.info('Loading parameters...')
        args = HParams(
            from_str=str(meta['training_args'])).update(vars(args_nondefault))

        epoch_offset = len(meta['epochs'])
        logger.info('Current epoch: %d' % epoch_offset)

        if args_nondefault.lr:
            logger.info('Setting current learning rate to %f...' % args.lr)
            K.set_value(model.optimizer.lr, args.lr)

    else:
        logger.info('Creating model...')
        # Recovering all valid models
        model_fn = utils.get_from_module('core.models', args.model)
        # Loading model
        model = model_fn(args.model_params)

        logger.info('Setting the optimizer...')
        # Optimization
        if args.opt == 'sgd':
            opt = SGD(lr=args.lr, momentum=args.momentum,
                      clipnorm=args.clipnorm)
        elif args.opt == 'adam':
            opt = Adam(lr=args.lr, clipnorm=args.clipnorm)

        # Compile with dummy loss
        model.compile(loss={'ctc': ctc_dummy_loss,
                            'decoder': decoder_dummy_loss},
                      optimizer=opt, metrics={'decoder': metrics.ler},
                      loss_weights=[1, 0])

    logger.info('Creating results folder...')
    # Creating the results folder
    name = args.save
    if name is None:
        name = os.path.join('results',
                            '%s_%s' % (args.model, datetime.datetime.now()))
    if not os.path.isdir(name):
        os.makedirs(name)

    logger.info('Adding callbacks')
    # Callbacks
    model_ckpt = MetaCheckpoint(os.path.join(name, 'model.h5'),
                                training_args=args, meta=meta)
    best_ckpt = MetaCheckpoint(
        os.path.join(name, 'best.h5'), monitor='val_decoder_ler',
        save_best_only=True, mode='min', training_args=args, meta=meta)
    callback_list = [model_ckpt, best_ckpt]

    # LR schedules
    if args.lr_schedule:
        lr_schedule_fn = utils.get_from_module('keras.callbacks',
                                               args.lr_schedule)
        if lr_schedule_fn:
            lr_schedule = lr_schedule_fn(
                *HParams(from_str=args.lr_params).values())
            callback_list.append(lr_schedule)

    logger.info('Getting the feature extractor...')
    # Features extractor
    feats_extractor = utils.get_from_module('preprocessing.audio',
                                            args.feats
                                            args.feats_params)

    logger.info('Getting the text parser...')
    # Recovering text parser
    text_parser = utils.get_from_module('preprocessing.text',
                                        args.text_parser,
                                        args.text_parser_params)

    logger.info('Getting the data generator...')
    # Data generator
    data_gen = DatasetGenerator(feats_extractor, text_parser,
                                batch_size=args.batch_size, seed=0)
    # iterators over datasets
    train_flow, valid_flow, test_flow = None, None, None

    logger.info('Generating flow...')
    if args.dataset is not None:
        train_flow, valid_flow, test_flow = data_gen.flows_from_fname(
            args.dataset)
    else:
        if args.train:
            train_flow = data_gen.flow_from_fname(args.train, dt_name='train')

        if args.valid:
            valid_flow = data_gen.flow_from_fname(args.valid, dt_name='valid')

        if args.test:
             test_flow = data_gen.flow_from_fname(args.test, dt_name='test')

    nb_val_samples = None
    if valid_flow:
        nb_val_samples = valid_flow.len

    logger.info('Initialzing training...')
    # Fit the model
    model.fit_generator(train_flow, samples_per_epoch=train_flow.len,
                        nb_epoch=args.nb_epoch, validation_data=valid_flow,
                        nb_val_samples=nb_val_samples, max_q_size=10,
                        nb_worker=1, callbacks=callback_list, verbose=1,
                        initial_epoch=epoch_offset)

    if test_flow:
        del model
        model = utils.load_model(os.path.join(name, 'best.h5'))
        logger.info('Evaluating best model on test set')
        metrics = model.evaluate_generator(test_flow, test_flow.len,
                                           max_q_size=10, nb_worker=1)

        msg = 'Total loss: %.4f\n\
CTC Loss: %.4f\nLER: %.2f' % (metrics[0], metrics[1], metrics[3])
        logger.info(msg)

        with open(os.path.join(name, 'results.txt'), 'w') as f:
            f.write(msg)
