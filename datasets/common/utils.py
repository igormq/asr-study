from __future__ import absolute_import, division, print_function

import h5py
import json
import librosa

import codecs
from unidecode import unidecode
import string

import argparse

import numpy as np
import time
import re
import os

from preprocessing.audio import LogFbank, MFCC
from datasets import DT_ABSPATH
from datasets.common import utils

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

def normalize_audio(input_list, mean, std):
    return [(i - mean)/std for i in input_list]
