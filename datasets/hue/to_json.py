# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import json

import numpy as np

import os
import re

import wave
import codecs

def read_from_json(json_file):
    dataset = []

    with codecs.open(json_file, 'r', encoding='utf8') as json_line_file:
        for line_num, json_line in enumerate(json_line_file):
            try:
                spec = json.loads(json_line, encoding='utf8')

                audio_file = spec['key']
                utterance = spec['text']

                if len(utterance) == 0:
                    raise ValueError("Utterance is empty")

                dataset.append([audio_file, utterance])

            except ValueError as e:
                print(u'Error reading line #{}: {}'.format(line_num, json_line))
                print(str(e))
    return dataset

def main(train_files, test_files):

    train_set = []
    for train_file in train_files:
        train_set.extend(read_from_json(train_file))

    test_set = []
    for test_file in test_files:
        test_set.extend(read_from_json(test_file))

    with codecs.open('train.json', 'w', encoding='utf8') as f:
        json.dump(train_set, f, ensure_ascii=False)

    with codecs.open('test.json', 'w', encoding='utf8') as f:
        json.dump(test_set, f, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', nargs='+', type=str, help='Location to json train files')
    parser.add_argument('--test_files', nargs='+', type=str, help='Location to json test files')
    args = parser.parse_args()

    main(train_files, test_files)
