from common.utils import load_meta
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print training arguments')
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()

    meta = load_meta(args.model)

    for k, v in meta['training_args'].items():
        print('%s: %s' % (k, v))
