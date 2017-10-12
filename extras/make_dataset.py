from __future__ import absolute_import, division, print_function

import argparse

from utils import generic_utils as utils
from utils.hparams import HParams

import preprocessing
import datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a preprocessed dataset (hdf5 file) by providing the path to the dataset and the correct parser.')

    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--parser', type=str, required=True)
    parser.add_argument('--parser_params', nargs='+', default=[])

    parser.add_argument('--output_file', type=str, default=None)

    parser.add_argument('--input_parser', type=str, default=None)
    parser.add_argument('--input_parser_params', nargs='+', default=[])

    parser.add_argument('--label_parser', type=str,
                        default=None)
    parser.add_argument('--label_parser_params', nargs='+', default=[])

    parser.add_argument('--override', action='store_true')

    args = parser.parse_args()

    parser = utils.get_from_module('datasets*',
                                   args.parser,
                                   regex=True)

    input_parser = utils.get_from_module('preprocessing.audio',
                                         args.input_parser,
                                         params=args.input_parser_params)
    label_parser = utils.get_from_module('preprocessing.text',
                                         args.label_parser,
                                         params=args.label_parser_params)

    dataset = parser(args.dataset_dir,
                     **HParams().parse(args.parser_params).values())

    output_file = dataset.to_h5(fname=args.output_file,
                                input_parser=input_parser,
                                label_parser=label_parser,
                                override=args.override)

    print('Dataset %s saved at %s' % (parser.name, output_file))
