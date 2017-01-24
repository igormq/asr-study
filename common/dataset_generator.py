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

    def __init__(self, feature_extractor=None,
                       text_parser=None, sr=16e3):
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

        batch_inputs, batch_seq_len = self._make_in(self.inputs[index_array])
        batch_labels = self._make_out(self.labels[index_array])

        return self._make_in_out(batch_inputs, batch_labels, seq_len=batch_seq_len)

    def _make_in_out(self, batch_inputs, batch_labels, seq_len=None):
        return [batch_inputs, batch_labels, batch_seq_len], [np.zeros((batch_inputs.shape[0],)), batch_labels]

    def _make_in(self, inputs):
        if self.feature_extractor is not None:
            inputs = np.asarray([self.feature_extractor(i) for i in inputs])

        batch_inputs = pad_sequences(inputs, dtype='float32', padding='post')
        batch_seq_len = np.asarray([i.shape[0] for i in inputs])
        return batch_inputs, batch_seq_len

    def _make_out(self, labels):
        if self.text_parser is not None:
            labels = np.asarray([self.text_parser(l) for l in labels])

        max_labels_size = np.max([len(l) for l in labels])
        batch_labels = scipy.sparse.lil_matrix((current_batch_size, max_labels_size), dtype='int32')
        for i, l in enumerate(labels):
             batch_labels[i, :l.size] = l
        return batch_labels

class H5Iterator(DatasetIterator):

    def __init__(self, h5group, batch_size=32, shuffle=False, seed=None):

        inputs = h5group['inputs']
        labels = h5group['labels']

        super(H5Iterator, self).__init__(inputs, labels, batch_size, shuffle, seed)

        self.num_feats = inputs.attrs['num_feats']
        self.durations = h5group['durations']

    def _make_in(self, inputs):
        inputs = [i.reshape(None, self.num_feats) for i in inputs]
        return super(H5Iterator, self)._make_in(inputs)
