from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import codecs
import json
import time

from preprocessing import audio, text
from common import utils
from common import apis

import speech_recognition as sr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating an ASR system \
over an API.')

    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--language', default='pt-BR', type=str)
    parser.add_argument('--all', action='store_true', help='Will evaluate \
over all dataset, not only with the dt key equals test.')

    # Label generation (if necessary)
    parser.add_argument('--text_parser', type=str,
                        default='simple_char_parser')
    parser.add_argument('--text_parser_params', type=str, default='{}')

    # Other configs
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--apis', default=['google', 'ibm', 'microsoft'],
                        nargs='+')

    args = parser.parse_args()

    # If save is not defined, it will use the folder name of dataset location
    save = args.save
    if args.save is None:
        save = '%s_eval_apis.json' % args.dataset.split(os.path.sep)[-2]

    # Recovering text parser
    text_parser = utils.get_from_module('preprocessing.text',
                                        args.text_parser,
                                        args.text_parser_params)

    if not utils.check_ext(args.dataset, 'json'):
        raise ValueError('dataset must be a json file')

    dataset = json.load(codecs.open(args.dataset, 'r', encoding='utf8'))

    if not args.all and 'dt' in dataset[0]:
        dataset = [d for d in dataset if d['dt'] == 'test']

    apis = {'google': apis.recognize_google,
            'ibm': apis.recognize_ibm,
            'microsoft': apis.recognize_bing}

    eval_apis = []
    if args.resume:
        with codecs.open(save, 'r', encoding='utf8') as f:
            eval_apis = json.load(f)

    for i, data in enumerate(dataset):

        if len(eval_apis) > i:
            result = eval_apis[i]
        else:
            result = {}
            result['label'] = data['label']
            result['audio'] = data['audio']

            if args.all and 'dt' in data:
                result['dt'] = data['dt']

        for api_name in args.apis:
            if api_name in result and result[api_name] != '':
                continue
            try:
                result[api_name] = apis[api_name](data['audio'], safe=False,
                                                  language=args.language)
            except Exception as e:
                result[api_name] = ''
                print(e)

        if len(eval_apis) > i:
            eval_apis[i] = result
        else:
            eval_apis.append(result)

        if (args.save_every % (i + 1)) == 0:
            with codecs.open(save, 'w', encoding='utf8') as f:
                json.dump(eval_apis, f)

        print('Done %d/%d' % (i + 1, len(dataset)))
        time.sleep(.1)

    with codecs.open(save, 'w', encoding='utf8') as f:
        json.dump(eval_apis, f)
