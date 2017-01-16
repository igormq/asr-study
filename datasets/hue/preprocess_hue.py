import pandas as pd
import numpy as np
import librosa
import h5py
import json

import codecs
from unidecode import unidecode
import string

import argparse

import time
import re

from keras.preprocessing.sequence import pad_sequences

def parse_str(text):
    text = unidecode(text).translate(string.maketrans("-'", '  '))
    return ' '.join(text.translate(None, string.punctuation).lower().split())

def preprocess_audio(audio_path, target_sr=8e3, eps=1e-9):
    '''
    Returns Features (time_steps, nb_features) and sequence length (scalar)
    '''

    y, sr = librosa.load(audio_path)

    y = librosa.core.resample(y, sr, target_sr)
    sr = target_sr

    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=int(1e-2*sr), n_fft=int(25e-3*sr), n_mels=40)
    d = librosa.feature.delta(S)
    dd = librosa.feature.delta(S, order=2)
    S_e = np.log(librosa.feature.rmse(S=S) + eps)
    d_e = np.log(librosa.feature.rmse(S=d) + eps)
    dd_e = np.log(librosa.feature.rmse(S=dd) + eps)
    return np.vstack((S, d, dd, S_e, d_e, dd_e)).T, S.shape[1]

def normalize_audio(input_list, mean, std):
    return [(i - mean)/std for i in input_list]

def get_char_map():
    dict_char = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}
    dict_char[' '] = len(dict_char)
    return dict_char

def preprocess_char(txt, dict_char, row, remove_punctuation=True):
    if remove_punctuation:
        txt = parse_str(txt)
    values = np.array([dict_char[c] for c in txt], dtype='int32')
    indices = np.hstack((row * np.ones((values.size,1)), np.arange(values.size)[:, None])).astype('int64')
    char_len = values.size
    return (values, indices, char_len)


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

    ignored_files = []
    with h5py.File('hue.h5', 'w') as h5_file:
        for file_list, name in zip([train_files, test_files], ['train', 'test']):
            t = time.time()

            input_list, seq_len_list = [], []
            char_values_list, char_indices_list, char_len_list = [], [], []

            row = 0
            for f in file_list:

                filepath, labels = f
                print('%s %d/%d: %s' % (name, row, len(file_list), filepath))

                # CHAR
                try:
                    char_values, char_indices, char_len =  preprocess_char(labels, dict_char, row)

                    if char_len == 0:
                        raise ValueError, 'Transcription is empty'
                except KeyError as e:
                    print('%s. Ignoring file %s' % (e, filepath))
                    ignored_files.append([name, filepath, labels])
                    continue
                except ValueError as e:
                    print('%s. Ignoring file %s' % (e, filepath))
                    ignored_files.append([name, filepath, labels])
                    continue
                # WAV
                features, seq_len = preprocess_audio(filepath)

                # WAV
                input_list.append(features)
                seq_len_list.append(seq_len)

                # CHAR
                char_values_list.append(char_values)
                char_indices_list.append(char_indices)
                char_len_list.append(char_len)

                row += 1

            # WAV
            if mean is None or std is None:
                input_stack = np.vstack(input_list)
                mean = input_stack.mean(axis=0)
                std = input_stack.std(axis = 0)
                # Do not normalize dims with very small std
                std[std < 1e-9] = 1.

            input_list = normalize_audio(input_list, mean, std)

            inputs = pad_sequences(input_list, dtype='float32', padding='post')
            seq_len = np.hstack(seq_len_list)

            # CHAR
            char_values = np.hstack(char_values_list)
            char_indices = np.vstack(char_indices_list)
            char_shape = np.array((len(char_len_list), np.max(char_len_list)))

            # Save to HDF5
            group = h5_file.create_group(name)

            # inputs
            inp_grp = group.create_group('inputs')
            inp_grp.create_dataset('data', data=inputs)
            inp_grp.create_dataset('seq_len', data=seq_len)

            # character labels
            char_grp = group.create_group('char')
            char_grp.create_dataset('values', data=char_values)
            char_grp.create_dataset('indices', data=char_indices)
            char_grp.create_dataset('shape', data=char_shape)

            print('Finished %s in %.fs.' %(name, time.time() - t))
            h5_file.flush()
    print('Generation complete.')

    with codecs.open('ignored_hue.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join([' '.join(l) for l in ignored_files]))
