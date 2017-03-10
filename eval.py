from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import json
import numpy as np
# Preventing pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import h5py
import inspect

from preprocessing import audio, text

from utils import generic_utils as utils
from utils.hparams import HParams

from datasets.dataset_generator import DatasetGenerator, DatasetIterator

from utils.core_utils import setup_gpu, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating an ASR system.')

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--subset', type=str, default='test')

    parser.add_argument('--batch_size', default=32, type=int)

    # Features generation (if necessary)
    parser.add_argument('--input_parser', type=str, default=None)
    parser.add_argument('--input_parser_params', nargs='+', default=[])

    # Label generation (if necessary)
    parser.add_argument('--label_parser', type=str,
                        default='simple_char_parser')
    parser.add_argument('--label_parser_params', nargs='+', default=[])

    # Other configs
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')

    parser.add_argument('--save_transcriptions', default=None, type=str)

    args = parser.parse_args()
    args_nondefault = utils.parse_nondefault_args(
        args, parser.parse_args(
            ['--model', args.model, '--dataset', args.dataset]))

    # GPU configuration
    setup_gpu(args.gpu, args.allow_growth)

    # Loading model
    model, meta = load_model(args.model, return_meta=True, mode='eval')

    args = HParams(**meta['training_args']).update(vars(args_nondefault))

    # Features extractor
    input_parser = utils.get_from_module('preprocessing.audio',
                                         args.input_parser,
                                         params=args.input_parser_params)

    # Recovering text parser
    label_parser = utils.get_from_module('preprocessing.text',
                                         args.label_parser,
                                         params=args.label_parser_params)

    data_gen = DatasetGenerator(input_parser, label_parser,
                                batch_size=args.batch_size, seed=0)
    test_flow = data_gen.flow_from_fname(args.dataset, datasets=args.subset)

    metrics = model.evaluate_generator(test_flow, test_flow.len,
                                       max_q_size=10, nb_worker=1)

    for m, v in zip(model.metrics_names, metrics):
        print('%s: %4f' % (m, v))

    from keras import backend as K; K.clear_session()
