import librosa
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import h5py
import time

def remove_ext(file_list):
    return [f[:-4] for f in file_list]

def preprocess_audio(audio_path):
    '''
    Returns Features (time_steps, nb_features) and sequence length (scalar)
    '''
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=int(1e-2*sr), n_fft=int(25e-3*sr), n_mels=40)
    d = librosa.feature.delta(S)
    dd = librosa.feature.delta(S, order=2)
    S_e = np.log(librosa.feature.rmse(S=S))
    d_e = np.log(librosa.feature.rmse(S=d))
    dd_e = np.log(librosa.feature.rmse(S=dd))
    return np.vstack((S, d, dd, S_e, d_e, dd_e)).T, S.shape[1]

def normalize_audio(input_list, mean, std):
    return [(i - mean)/std for i in input_list]

def get_char_map():
    dict_char = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}
    dict_char[' '] = len(dict_char)
    dict_char["'"] = len(dict_char)
    dict_char['"'] = len(dict_char)
    dict_char['-'] = len(dict_char)
    dict_char[','] = len(dict_char)
    dict_char['.'] = len(dict_char)
    dict_char['!'] = len(dict_char)
    dict_char['?'] = len(dict_char)
    dict_char[':'] = len(dict_char)
    dict_char[';'] = len(dict_char)
    return dict_char

def preprocess_char(transcript_path, dict_char, row):
    with open(transcript_path, 'rb') as f:
        txt = f.readlines()[0]
    txt = ' '.join(txt.strip().split(' ')[2:]).lower()
    values = np.array([dict_char[c] for c in txt], dtype='int32')
    indices = np.hstack((row * np.ones((values.size,1)), np.arange(values.size)[:, None])).astype('int64')
    char_len = values.size
    return (values, indices, char_len)


def get_phn_map(map_path):
    with open(map_path, 'rb') as f:
        txt = f.readlines()
    phn_list = [l.strip().split('\t') for l in txt]
    phn_map = {p[0]:p[2] for p in phn_list if len(p) == 3}
    dict_phn = {k: v for v, k in enumerate(set(phn_map.values()))}
    return phn_map, dict_phn

def preprocess_phn(phn_path, phn_map, dict_phn, row):
    with open(phn_path, 'rb') as f:
        txt = f.readlines()
    phn_list = [(l.strip().split(' ')[2]).lower() for l in txt]
    phn_list = [phn_map[p] for p in phn_list if p in phn_map.keys()]
    values = np.array([dict_phn[p] for p in phn_list], dtype='int32')
    indices = np.hstack((row * np.ones((values.size,1)), np.arange(values.size)[:, None])).astype('int64')
    phn_len = values.size
    return (values, indices, phn_len)


if __name__ == '__main__':
    dataset_dir = 'timit'

    files = librosa.util.find_files(dataset_dir)

    # Ignoring SA sentences
    files = [f for f in files if re.search(r"SA([0-9])\.WAV", f) is None]

    train_files = [f for f in files if f.find('TRAIN') + 1]
    test_valid_files = [f for f in files if f.find('TEST') + 1]

    test_spk_list = [s.strip().upper() for s in open('timit/test_spk.list', 'rb').readlines()]
    test_files = [f for f in test_valid_files if any(s in f for s in test_spk_list)]

    valid_spk_list = [s.strip().upper() for s in open('timit/dev_spk.list', 'rb').readlines()]
    valid_files = [f for f in test_valid_files if any(s in f for s in valid_spk_list)]

    # Remove extension
    train_files = remove_ext(train_files)
    test_files = remove_ext(test_files)
    valid_files = remove_ext(valid_files)

    print('Got all file names.')

    dict_char = get_char_map()
    phn_map, dict_phn = get_phn_map('timit/phones.60-48-39.map')

    print('Got the dictionaries.')

    print('Generating Tensors...')
    mean, std = None, None
    with h5py.File('timit.h5', 'w') as h5_file:
        for file_list, name in zip([train_files, valid_files, test_files], ['train', 'valid', 'test']):
            t = time.time() # measure time

            input_list, seq_len_list = [], []
            char_values_list, char_indices_list, char_len_list = [], [], []
            phn_values_list, phn_indices_list, phn_len_list = [], [], []
            for row, f in enumerate(file_list):
                # WAV
                features, seq_len = preprocess_audio(f + '.WAV')
                input_list.append(features)
                seq_len_list.append(seq_len)

                # CHAR
                char_values, char_indices, char_len =  preprocess_char(f + '.TXT', dict_char, row)
                char_values_list.append(char_values)
                char_indices_list.append(char_indices)
                char_len_list.append(char_len)

                # PHN
                phn_values, phn_indices, phn_len = preprocess_phn(f + '.PHN', phn_map, dict_phn, row)
                phn_values_list.append(phn_values)
                phn_indices_list.append(phn_indices)
                phn_len_list.append(phn_len)

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
            char_shape = np.array((len(file_list), np.max(char_len_list)))

            # PHN
            phn_values = np.hstack(phn_values_list)
            phn_indices = np.vstack(phn_indices_list)
            phn_shape = np.array((len(file_list), np.max(phn_len_list)))

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

            # phoneme labels
            phn_grp = group.create_group('phn')
            phn_grp.create_dataset('values', data=phn_values)
            phn_grp.create_dataset('indices', data=phn_indices)
            phn_grp.create_dataset('shape', data=phn_shape)

            print('Finished %s in %.fs.' %(name, time.time() - t))
            h5_file.flush()
    print('Generation complete.')
