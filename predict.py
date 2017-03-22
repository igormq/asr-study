import argparse
import json
import numpy as np
import codecs

from datasets.dataset_generator import DatasetGenerator, DatasetIterator

from utils.core_utils import setup_gpu, load_model

from utils.hparams import HParams
from utils import generic_utils as utils

from preprocessing import audio, text

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluating an ASR system.')

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--file', default=None, type=str)
    parser.add_argument('--subset', type=str, default='test')

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

    parser.add_argument('--save', default=None, type=str)

    args = parser.parse_args()
    args_nondefault = utils.parse_nondefault_args(
        args, parser.parse_args(
            ['--model', args.model, '--dataset', args.dataset]))

    if args.dataset is None and args.file is None:
        raise ValueError('dataset or file args must be set.')

    if args.dataset and args.file:
        print('Both dataset and file args was set. Ignoring file args.')

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

    if args.dataset is not None:
        data_gen = DatasetGenerator(input_parser, label_parser,
                                    batch_size=1, seed=0, mode='predict')
        test_flow = data_gen.flow_from_fname(args.dataset, datasets=args.subset)
    else:
        test_flow = DatasetIterator(np.array([args.file]), None,
                                    input_parser=input_parser,
                                    label_parser=label_parser, mode='predict')
        test_flow.labels = np.array([u''])

    model = load_model(args.model, mode='predict')

    results = []
    for index in range(test_flow.len):
        prediction = model.predict(test_flow.next())
        prediction = label_parser.imap(prediction[0])
        results.append({'label': test_flow.labels[0], 'best': prediction})
        print('Ground Truth: %s' % (label_parser._sanitize(test_flow.labels[0])))
        print('   Predicted: %s\n\n' % prediction)

    if args.save is not None:
        with codecs.open(args.save, 'w', encoding='utf8') as f:
            json.dump(results, f)

    from keras import backend as K; K.clear_session()
