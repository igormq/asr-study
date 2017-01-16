import numpy as np
import librosa
import json

import codecs
from unidecode import unidecode
import string

import argparse

import time
import re

import tensorflow as tf

from common.sequence_example_lib import make_sequence_example

def parse_str(text):
    text = unidecode(text).translate(string.maketrans("-'", '  '))
    return ' '.join(text.translate(None, string.punctuation).lower().split())

def preprocess_char(txt, dict_char, remove_punctuation=True):
    if remove_punctuation:
        txt = parse_str(txt)
    label = np.array([dict_char[c] for c in txt], dtype='int32')

    return label

def get_data(file_list, target_sr=8e3, eps=1e-9):
    for row, (audio_path, label) in enumerate(file_list):

        print('%s %d/%d: %s' % (dt_name, row, len(file_list), audio_path))

        # WAV
        audio, sr = librosa.load(audio_path)
        audio = librosa.core.resample(audio, sr, target_sr)

        # Normalizing audio
        audio = (audio - np.mean(audio)) / (np.std(audio) + eps)

        yield(audio, label)

def get_char_map():
    dict_char = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}
    dict_char[' '] = len(dict_char)
    return dict_char


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        help='Path to train.json')
    parser.add_argument('--test', type=str,
                        help='Path to test.json')

    args = parser.parse_args()

    train_files = json.load(codecs.open(args.train, 'r', encoding='utf8'))
    test_files = json.load(codecs.open(args.test, 'r', encoding='utf8'))
    print('Got all filenames.')

    dict_char = get_char_map()
    print('Got the dictionaries.')

    print('Generating Tensors...')
    mean, std = None, None

    global_time = time.time()
    for file_list, dt_name in zip([train_files, test_files], ['train', 'test']):
        t = time.time()

        writer = tf.python_io.TFRecordWriter('%s.tfrecord' % (dt_name))

        for sequence, label in get_data(file_list):

            # CHAR
            try:
                label =  preprocess_char(label, dict_char)
            except (KeyError, ValueError) as e:
                print('%s. Ignoring line: %s' % (e, label))
                continue

            ex = make_sequence_example(sequence, label)
            writer.write(ex.SerializeToString())

        writer.close()

        print('Finished %s in %.fs.' %(dt_name, time.time() - t))
    print('Generation complete. Total elapsed time %.fs' % (time.time() - global_time))
