from __future__ import absolute_import, division, print_function

import argparse

from utils.core_utils import load_meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print training arguments')
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()

    meta = load_meta(args.model)

    for k, v in meta['training_args'].items():
        print('%s: %s' % (k, v))
