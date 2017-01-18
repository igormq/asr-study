from __future__ import absolute_import, division, print_function

import sys
sys.path.append('/home/igormq/repos/icassp-2017/')

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

from keras.preprocessing.sequence import pad_sequences

from preprocessing.audio import LogFbank, MFCC
from datasets import DT_ABSPATH
from datasets.common import utils

def preprocess_audio(audio_path, transform, target_sr=8e3, eps=1e-9):
    '''
    Returns Features (time_steps, nb_features) and sequence length (scalar)
    '''

    y, sr = librosa.load(audio_path)

    y = librosa.core.resample(y, sr, target_sr)
    sr = target_sr

    S = transform(y)

    return S, S.shape[0]

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


def main(output_path):

    if not os.path.exists(os.path.join(output_path, 'data.json')):
        raise IOError, "JSON file not found for this dataset"

    # Features extractor
    logfbank = LogFbank(d=True, dd=True, append_energy=True)
    mfcc = MFCC(d=True, dd=True, append_energy=True)

    data_files = read_from_json(os.path.join(output_path, 'data.json'))
    print('Got all filenames.')

    print('Generating Tensors...')
    mean_mfcc, std_mfcc = None, None
    mean_fbank, std_fbank = None, None

    ignored_files = []
    with h5py.File(os.path.join(output_path, 'data.h5'), 'w') as h5_file:
        for file_list, name in zip([data_files], ['train']):
            t = time.time()

            input_list_mfcc, seq_len_list_mfcc = [], []
            input_list_fbank, seq_len_list_fbank = [], []

            labels_list = []

            for row, f in enumerate(file_list):

                filepath, labels = f
                print('%s %d/%d: %s' % (name, row, len(file_list), filepath))

                # CHAR
                try:
                    if len(labels) == 0:
                        raise ValueError, 'Transcription is empty'
                except KeyError as e:
                    print('%s. Ignoring file %s' % (e, filepath))
                    ignored_files.append([name, filepath, labels])
                    continue
                except ValueError as e:
                    print('%s. Ignoring file %s' % (e, filepath))
                    ignored_files.append([name, filepath, labels])
                    continue

                # MFCC
                features_mfcc, seq_len_mfcc = preprocess_audio(filepath, mfcc)
                input_list_mfcc.append(features_mfcc)
                seq_len_list_mfcc.append(seq_len_mfcc)

                # LogFbank
                features_fbank, seq_len_fbank = preprocess_audio(filepath, logfbank)
                input_list_fbank.append(features_fbank)
                seq_len_list_fbank.append(seq_len_fbank)

                # CHAR
                labels_list.append(labels)

            # Save to HDF5
            group = h5_file

            # WAV
            if mean_mfcc is None or std_mfcc is None:
                input_stack = np.vstack(input_list_mfcc)
                mean_mfcc = input_stack.mean(axis=0)
                std_mfcc = input_stack.std(axis = 0)
                # Do not normalize dims with very small std
                std_mfcc[std_mfcc < 1e-9] = 1.

            if mean_fbank is None or std_fbank is None:
                input_stack = np.vstack(input_list_fbank)
                mean_fbank = input_stack.mean(axis=0)
                std_fbank = input_stack.std(axis = 0)
                # Do not normalize dims with very small std
                std_fbank[std_fbank < 1e-9] = 1.

            # input_list_mfcc = normalize_audio(input_list_mfcc, mean_mfcc, std_mfcc)
            # input_list_fbank = normalize_audio(input_list_fbank, mean_fbank, std_fbank)

            inputs_mfcc = pad_sequences(input_list_mfcc, dtype='float32', padding='post')
            seq_len_mfcc = np.hstack(seq_len_list_mfcc)

            inputs_fbank = pad_sequences(input_list_fbank, dtype='float32', padding='post')
            seq_len_fbank = np.hstack(seq_len_list_fbank)

            # inputs mfcc
            inp_grp = group.create_group('mfcc')
            inp_grp.create_dataset('data', data=inputs_mfcc)
            inp_grp.create_dataset('seq_len', data=seq_len_mfcc)
            inp_grp.attrs['mean'] = mean_mfcc
            inp_grp.attrs['std'] = std_mfcc

            # inputs fbank
            inp_grp = group.create_group('fbank')
            inp_grp.create_dataset('data', data=inputs_fbank)
            inp_grp.create_dataset('seq_len', data=seq_len_fbank)
            inp_grp.attrs['mean'] = mean_fbank
            inp_grp.attrs['std'] = std_fbank

            # character labels
            group.create_dataset('labels', data=np.asarray([l.encode('utf8') for l in labels]))

            print('Finished %s in %.fs.' %(name, time.time() - t))
            h5_file.flush()

    print('Generation complete.')

    with codecs.open(os.path.join(output_path, 'ignored.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join([' '.join(l) for l in ignored_files]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str,
                        help='dataset name', required=True)
    args = parser.parse_args()

    main(os.path.join(DT_ABSPATH, args.dt))
