from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Preventing pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import uuid
import sys
import json
import datetime
import inspect
import codecs

import logging
try:
    import warpctc_tensorflow
except ImportError:
    logging.warning('warpctc binding for tensorflow not found. :(')
import tensorflow as tf

import keras

import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau

from core import metrics
from core.ctc_utils import ctc_dummy_loss, decoder_dummy_loss
from core.callbacks import MetaCheckpoint, ProgbarLogger
from utils.core_utils import setup_gpu

from preprocessing import audio, text

from datasets.dataset_generator import DatasetGenerator
from utils.hparams import HParams

import utils.generic_utils as utils

from utils.core_utils import load_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training an ASR system.')

    # Resume training
    parser.add_argument('--load', default=None, type=str)

    # Model settings
    parser.add_argument('--model', default='brsmv1', type=str)
    parser.add_argument('--model_params', nargs='+', default=[])

    # Hyper parameters
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clipnorm', default=400, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['sgd', 'adam'])
    # End of hyper parameters

    # Dataset definitions
    parser.add_argument('--dataset', default=None, type=str, nargs='+')

    # Features generation (if necessary)
    parser.add_argument('--input_parser', type=str, default=None)
    parser.add_argument('--input_parser_params', nargs='+', default=[])

    # Label generation (if necessary)
    parser.add_argument('--label_parser', type=str,
                        default='simple_char_parser')
    parser.add_argument('--label_parser_params', nargs='+', default=[])

    # Callbacks
    parser.add_argument('--lr_schedule', default=None)
    parser.add_argument('--lr_params', nargs='+', default=[])

    # Other configs
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--seed', default=None, type=float)

    args = parser.parse_args()

    # Setup logging
    utils.setup_logging()
    logger = logging.getLogger(__name__)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # hack in ProgbarLogger: avoid logger.infoing the dummy losses
    keras.callbacks.ProgbarLogger = lambda: ProgbarLogger(
        show_metrics=['loss', 'decoder_ler', 'val_loss', 'val_decoder_ler'])

    # GPU configuration
    setup_gpu(args.gpu, args.allow_growth,
              log_device_placement=args.verbose > 1)

    # Initial configuration
    epoch_offset = 0
    meta = None

    if args.load:
        args_nondefault = utils.parse_nondefault_args(args,
                                                      parser.parse_args([]))

        logger.info('Loading model...')
        model, meta = load_model(args.load, return_meta=True)

        logger.info('Loading parameters...')
        args = HParams(**meta['training_args']).update(vars(args_nondefault))

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
        model = model_fn(**(HParams().parse(args.model_params).values()))

        logger.info('Setting the optimizer...')
        # Optimization
        if args.opt.strip().lower() == 'sgd':
            opt = SGD(lr=args.lr, momentum=args.momentum,
                      clipnorm=args.clipnorm)
        elif args.opt.strip().lower() == 'adam':
            opt = Adam(lr=args.lr, clipnorm=args.clipnorm)

        # Compile with dummy loss
        model.compile(loss={'ctc': ctc_dummy_loss,
                            'decoder': decoder_dummy_loss},
                      optimizer=opt, metrics={'decoder': metrics.ler},
                      loss_weights=[1, 0])

    logger.info('Creating results folder...')
    # Creating the results folder
    output_dir = args.save
    if output_dir is None:
        output_dir = os.path.join('results',
                                  '%s_%s' % (args.model,
                                             datetime.datetime.now()))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger.info('Adding callbacks')
    # Callbacks
    model_ckpt = MetaCheckpoint(os.path.join(output_dir, 'model.h5'),
                                training_args=args, meta=meta)
    best_ckpt = MetaCheckpoint(
        os.path.join(output_dir, 'best.h5'), monitor='val_decoder_ler',
        save_best_only=True, mode='min', training_args=args, meta=meta)
    callback_list = [model_ckpt, best_ckpt]

    # LR schedules
    if args.lr_schedule:
        lr_schedule_fn = utils.get_from_module('keras.callbacks',
                                               args.lr_schedule)
        if lr_schedule_fn:
            lr_schedule = lr_schedule_fn(**HParams().parse(args.lr_params).values())
            callback_list.append(lr_schedule)
        else:
            raise ValueError('Learning rate schedule unrecognized')

    logger.info('Getting the feature extractor...')
    # Features extractor
    input_parser = utils.get_from_module('preprocessing.audio',
                                         args.input_parser,
                                         params=args.input_parser_params)

    logger.info('Getting the text parser...')
    # Recovering text parser
    label_parser = utils.get_from_module('preprocessing.text',
                                         args.label_parser,
                                         params=args.label_parser_params)

    logger.info('Getting the data generator...')
    # Data generator
    data_gen = DatasetGenerator(input_parser, label_parser,
                                batch_size=args.batch_size,
                                seed=args.seed)
    # iterators over datasets
    train_flow, valid_flow, test_flow = None, None, None
    num_val_samples = num_test_samples = 0

    logger.info('Generating flow...')
    if len(args.dataset) == 1:
        train_flow, valid_flow, test_flow = data_gen.flow_from_fname(
            args.dataset[0], datasets=['train', 'valid', 'test'])
        num_val_samples = valid_flow.len
    else:
        train_flow = data_gen.flow_from_fname(args.dataset[0])
        valid_flow = data_gen.flow_from_fname(args.dataset[1])

        num_val_samples = valid_flow.len
        if len(args.dataset) == 3:
            test_flow = data_gen.flow_from_fname(args.dataset[2])
            num_test_samples = test_flow.len

    logger.info(str(vars(args)))
    print(str(vars(args)))
    logger.info('Initialzing training...')
    # Fit the model
    model.fit_generator(train_flow, samples_per_epoch=train_flow.len,
                        nb_epoch=args.num_epochs, validation_data=valid_flow,
                        nb_val_samples=num_val_samples, max_q_size=10,
                        nb_worker=1, callbacks=callback_list, verbose=1,
                        initial_epoch=epoch_offset)

    if test_flow:
        del model
        model = load_model(os.path.join(output_dir, 'best.h5'), mode='eval')
        logger.info('Evaluating best model on test set')
        metrics = model.evaluate_generator(test_flow, test_flow.len,
                                           max_q_size=10, nb_worker=1)

        msg = 'Total loss: %.4f\n\
CTC Loss: %.4f\nLER: %.2f%%' % (metrics[0], metrics[1], metrics[3]*100)
        logger.info(msg)

        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write(msg)

        print(msg)

    K.clear_session()
