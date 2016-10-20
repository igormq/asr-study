import librosa
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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

dict_char = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}
dict_char[' '] = len(dict_char)
dict_char["'"] = len(dict_char)
dict_char['"'] = len(dict_char)
dict_char['-'] = len(dict_char)
dict_char[','] = len(dict_char)
dict_char['.'] = len(dict_char)
dict_char['?'] = len(dict_char)
dict_char[':'] = len(dict_char)
dict_char[';'] = len(dict_char)

def preprocess_char(transcript_path, row):
    with open(transcript_path, 'rb') as f:
        txt = f.readlines()[0]
    txt = ' '.join(txt.strip().split(' ')[2:]).lower()
    values = np.array([dict_char[c] for c in txt], dtype='int32')
    indices = np.hstack((row * np.ones((values.size,1)), np.arange(values.size)[:, None])).astype('int64')
    char_len = values.size
    return (values, indices, char_len)

def preprocess_phn(phn_path):
    with open(phn_path, 'rb') as f:
        txt = f.readlines()
    phn_list = [t.strip().split(' ')[2] for t in txt]


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
print('Generating Tensors...')
for file_list in [train_files, test_files, valid_files]:


    # input_list = []
    # seq_len_list = []
    char_values_list, char_indices_list, char_len_list = [], [], []
    for row, f in enumerate(file_list):
        # WAV
    #     features, seq_len = preprocess_audio(f + '.WAV')
    #     input_list.append(features)
    #     seq_len_list.append(seq_len)

        # CHAR
        char_values, char_indices, char_len =  preprocess_char(f + '.TXT', row)
        char_values_list.append(char_values)
        char_indices_list.append(char_indices)
        char_len_list.append(char_len)
    #
    # X = pad_sequences(input_list, dtype='float32', padding='post')

    # CHAR
    char_values = np.hstack(char_values_list)
    char_indices = np.vstack(char_indices_list)
    char_shape = np.array((len(file_list), np.max(char_len_list)))
    print('Finished one.')
print('Generation complete.')
