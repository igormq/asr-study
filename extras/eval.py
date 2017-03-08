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
from common import utils
from common.dataset_generator import DatasetGenerator, DatasetIterator
from common.hparams import HParams
from core.utils import setup_gpu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating an ASR system.')

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    # Features generation (if necessary)
    parser.add_argument('--feats', type=str, default='raw',
                        choices=['mfcc', 'raw', 'logfbank'])
    parser.add_argument('--feats_params', type=str, default='{}')

    # Label generation (if necessary)
    parser.add_argument('--text_parser', type=str,
                        default='simple_char_parser')
    parser.add_argument('--text_parser_params', type=str, default='{}')

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
    model, meta = utils.load_model(args.model, return_meta=True, mode='eval')

    args = HParams(
        from_str=str(meta['training_args'])).update(vars(args_nondefault))

    # Features extractor
    feats_extractor = utils.get_from_module('preprocessing.audio',
                                            args.feats,
                                            args.feats_params)

    # Recovering text parser
    text_parser = utils.get_from_module('preprocessing.text',
                                        args.text_parser,
                                        args.text_parser_params)

    data_gen = DatasetGenerator(feats_extractor, text_parser,
                                batch_size=args.batch_size, seed=0)
    test_flow = data_gen.flow_from_fname(args.dataset, dt_name='test')

    metrics = model.evaluate_generator(test_flow, test_flow.len,
                                       max_q_size=10, num_worker=1)

    for m, v in zip(model.metrics_names, metrics):
        print('%s: %4f' % (m, v))

    if args.save_transcriptions:
        del model
        data = h5py.File(args.dataset)

        model_p = utils.load_model(args.model, mode='predict')
        results = []
        for i in range(len(data['mfcc']['test']['inputs'])):
            data_it = DatasetIterator(np.array(
                [np.array(data['mfcc']['test']['inputs'][i])
                 .reshape((-1, 39))]))
            label = data[str(feats_extractor)]['test']['labels'][i]
            prediction = model_p.predict(data_it.next())
            prediction = text_parser.imap(prediction[0])
            results.append({'label': label, 'best': prediction})

        with codecs.open(args.save_transcriptions, 'w', encoding='utf8') as f:
            json.dump(results, f)
