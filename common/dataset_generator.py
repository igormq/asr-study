from __future__ import absolute_import, division, print_function

from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences

import scipy
import librosa
import numpy as np

from preprocessing import audio, text

class DatasetGenerator(object):
    ''' Generates mini-batches from json file
    '''

    def __init__(self, feature_extractor=audio.raw,
                       text_parser=text.simple_parser, sr=16e3):
        self.sr = sr
        self.feature_extractor = feature_extractor
        self.text_parser = text_parser


    def flow_from_json(self, json_file, batch_size=32, shuffle=True, seed=None):
        raise NotImplementedError

    def flow(self, audio_paths, texts, batch_size=32, shuffle=True, seed=None):
        return DatasetIterator(audio_paths, texts, self, batch_size, shuffle, seed)

class DatasetIterator(Iterator):

    def __init__(self, audio_paths, texts, audio_gen,
                 batch_size=32, shuffle=False, seed=None):

        if texts is not None and len(audio_paths) != len(texts):
            raise ValueError('audio_paths texts '
                             'should have the same length. '
                             'Found: len(audio_paths) = %s, len(texts) = %s' % (len(audio_paths), len(texts)))

        self.audio_paths = audio_paths
        self.texts = texts
        self.audio_gen = audio_gen

        super(DatasetIterator, self).__init__(len(audio_paths), batch_size, shuffle, seed)


    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_inputs, batch_labels, batch_ind, batch_len, batch_seq_len = [], [], [], [], []

        for i, j in enumerate(index_array):
            audio_path = self.audio_paths[j]
            txt = self.texts[j]

            audio, _ = librosa.load(audio_path, self.audio_gen.sr)

            input_ = self.audio_gen.feature_extractor(audio)

            label = np.array(self.audio_gen.text_parser(txt), dtype='int32')
            indices = np.hstack((i * np.ones((label.size,1)), np.arange(label.size)[:, None])).astype('int64')
            label_len = label.size

            batch_inputs.append(input_)
            batch_seq_len.append(input_.shape[0])

            batch_labels.append(label)
            batch_ind.append(indices)
            batch_len.append(label_len)


        batch_inputs = pad_sequences(batch_inputs, dtype='float32', padding='post')

        batch_labels = np.hstack(batch_labels)

        batch_ind = np.vstack(batch_ind)
        batch_ind = (batch_ind[:, 0], batch_ind[:, 1])

        batch_labels = scipy.sparse.coo_matrix((batch_labels, batch_ind), shape=np.array((len(batch_len), np.max(batch_len)))).tolil()

        batch_seq_len = np.hstack(batch_seq_len)


        return [batch_inputs, batch_labels, batch_seq_len], [np.zeros((batch_inputs.shape[0],)), batch_labels]
