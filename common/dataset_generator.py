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
                       text_parser=text.simple_char_parser, sr=16e3):
        self.sr = sr
        self.feature_extractor = feature_extractor
        self.text_parser = text_parser

    def flow_from_json(self, json_file, batch_size=32, shuffle=True, seed=None):
        raise NotImplementedError

    def flow_from_h5(self, h5_group, batch_size=32, shuffle=True, seed=None):
        raise NotImplementedError

    def flow(self, inputs, labels, batch_size=32, shuffle=True, seed=None):
        return DatasetIterator(inputs, labels, batch_size, shuffle, seed)

class DatasetIterator(Iterator):

    def __init__(self, inputs, labels, batch_size=32, shuffle=False, seed=None):

        if labels is not None and len(inputs) != len(labels):
            raise ValueError('inputs and labels '
                             'should have the same length. '
                             'Found: len(inputs) = %s, len(labels) = %s' % (len(inputs), len(labels)))

        self.inputs = inputs
        self.labels = labels

        super(DatasetIterator, self).__init__(len(inputs), batch_size, shuffle, seed)


    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        max_labels_size = np.max([len(l) for l in self.labels[index_array]])
        batch_labels = scipy.sparse.lil_matrix((current_batch_size, max_labels_size), dtype='int32')
        for i, l in enumerate(self.labels[index_array]):
             batch_labels[i, :l.size] = l

        batch_inputs = pad_sequences(self.inputs[index_array], dtype='float32', padding='post')
        batch_seq_len = np.asarray([i.shape[0] for i in self.inputs[index_array]])


        return [batch_inputs, batch_labels, batch_seq_len], [np.zeros((batch_inputs.shape[0],)), batch_labels]
