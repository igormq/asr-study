import argparse
import json
import h5py
import os
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
    parser.add_argument('--no_decoder', action='store_true', default=False)

    # Other configs
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')

    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--override', default=False, action='store_true')

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
    model, meta = load_model(args.model, return_meta=True,
                             mode='predict', decoder=(not args.no_decoder))

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
                                    batch_size=1, seed=0, mode='predict',
                                    shuffle=False)
        test_flow = data_gen.flow_from_fname(args.dataset,
                                             datasets=args.subset)
    else:
        test_flow = DatasetIterator(np.array([args.file]), None,
                                    input_parser=input_parser,
                                    label_parser=label_parser, mode='predict',
                                    shuffle=False)
        test_flow.labels = np.array([u''])

    results = []
    for index in range(test_flow.len):
        prediction = model.predict(test_flow.next())
        if not args.no_decoder:
            prediction = label_parser.imap(prediction[0])
        results.append({'input': test_flow.inputs[0].tolist(), 'label': test_flow.labels[0], 'best': prediction.tolist()})
        print('Ground Truth: %s' % (label_parser._sanitize(test_flow.labels[0])))
        print('   Predicted: %s\n\n' % prediction)

    if args.save is not None:
        if os.path.exists(args.save):
            if not args.override:
                raise IOError('Unable to create file')
            os.remove(args.save)

        if args.no_decoder:
            with h5py.File(args.save) as f:
                predictions = f.create_dataset(
                    'predictions', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=np.dtype('float32')))
                predictions.attrs['num_labels'] = results[0]['prediction'].shape[-1]

                labels = f.create_dataset(
                    'labels', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=unicode))

                inputs = f.create_dataset(
                    'inputs', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=unicode))

                for index, result in enumerate(results):

                    label = result['label']
                    prediction = result['prediction']
                    input_ = result['input']

                    inputs.resize(inputs.shape[0] + 1, axis=0)
                    inputs[inputs.shape[0] - 1] = input_

                    labels.resize(labels.shape[0] + 1, axis=0)
                    labels[labels.shape[0] - 1] = label.encode('utf8')

                    predictions.resize(predictions.shape[0] + 1, axis=0)
                    predictions[predictions.shape[0] - 1] = prediction.flatten().astype('float32')

                    # Flush to disk only when it reaches 128 samples
                    if index % 128 == 0:
                        print('%d/%d done.' % (index, len(results)))
                        f.flush()

                f.flush()
                print('%d/%d done.' % (len(results), len(results)))
        else:
            raise ValueError('save param must be set if no_decoder is Truepython')

    else:
        with codecs.open(args.save, 'w', encoding='utf8') as f:
            json.dump(results, f)

    from keras import backend as K
    K.clear_session()
