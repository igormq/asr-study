# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
# Import CTC loss
try:
    from tensorflow.contrib.warpctc.ctc_ops import warp_ctc_loss as ctc_loss
except ImportError:
    from tensorflow.python.ops.ctc_ops import ctc_loss

import numpy as np
import librosa
from scipy import sparse

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda, TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from layers import RHN

def ctc_lambda_func(args):
    y_pred, labels, input_length = args
    return ctc_loss(tf.transpose(y_pred, perm=[1, 0, 2]), tf.sparse_reorder(labels), input_length[:, 0])

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=int(1e-2*sr), n_fft=int(25e-3*sr), n_mels=40)
    d = librosa.feature.delta(S)
    dd = librosa.feature.delta(S, order=2)
    S_e = np.log(librosa.feature.rmse(S=S))
    d_e = np.log(librosa.feature.rmse(S=d))
    dd_e = np.log(librosa.feature.rmse(S=dd))
    return np.vstack((S, d, dd, S_e, d_e, dd_e)).T[None, :], np.array([S.shape[1]])

dict_char = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}
dict_char[' '] = len(dict_char)
dict_char['.'] = len(dict_char)

def preprocess_transcript(transcript_path):
    with open(transcript_path, 'rb') as f:
        txt = f.readlines()[0]
    txt = ' '.join(txt.strip().split(' ')[2:]).lower()
    values = np.array([dict_char[c] for c in txt], dtype='int32')
    # indices = np.hstack((np.zeros((values.size,1)), np.arange(values.size)[:, None])).astype('int64')
    indices = (np.zeros((values.size)), np.arange(values.size))
    shape = np.array([1, values.size], dtype='int64')
    return sparse.coo_matrix((values, indices), shape=shape).tolil()

inv_dict = {v: k for (k, v) in dict_char.iteritems()}
inv_dict[len(inv_dict)] = 'B'
def get_output(model, x):
    return ''.join([inv_dict[i] for i in model.predict(x).argmax(axis=2)[0]])

if __name__ == '__main__':

    audio_path = get_file('LDC93S1.wav', origin='https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav')
    transcript_path = get_file('LDC93S1.txt', origin='https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.txt')

    X, X_length = preprocess_audio(audio_path)
    y = preprocess_transcript(transcript_path)

    nb_features = X.shape[2]
    nb_classes = len(dict_char) + 1 # Blank

    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")

    # Define placeholders
    x = Input(name='input', shape=(None, nb_features))
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    input_length = Input(name='input_length', shape=(None,), dtype='int32')

    # Define model
    o = RHN(250, return_sequences=True)(x)
    o = TimeDistributed(Dense(nb_classes))(o)
    # Define loss as a layer
    l = ctc([o, labels, input_length])

    model = Model(input=[x, labels, input_length], output=l)
    pred = Model(input=x, output=o)

    # Optimization
    opt = RMSprop(lr=0.001, clipnorm=10)
    # Compile with dummy loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

    # FITZERA PESADAO
    # model.fit([X, y, X_length], np.zeros((1,)), batch_size=1, nb_epoch=500)
