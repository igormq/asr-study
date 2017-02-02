from __future__ import absolute_import, division, print_function

from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences

import scipy
import librosa
import h5py
import numpy as np
import codecs
import json
import os

from preprocessing import audio, text

from . import utils


class DatasetGenerator(object):
    """ Dataset generator that handles several forms of input and return an
    iterator over it. Only works for a CTC model

    # Arguments
        feature_extractor: instance of Feature [preprocessing.audio.Feature]
            feature that is applied to each audio file (or audio data)
        text_parser: instance of Parser [preprocessing.text.Parser].
            parser that is applied to each label data
        batch_size: number of samples per batch
        shuffle: reordering index per epoch. This avoid some bias in training
        seed: default None
    """

    def __init__(self, feature_extractor=None, text_parser=None, batch_size=32,
                 shuffle=True, seed=None):
        self.feature_extractor = feature_extractor
        self.text_parser = text_parser
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def flow_from_fname(self, fname, dt_name=None):
        """ Returns an specific iterator given the filename

        # Arguments
            dt_name: returns an iterator over an specific dataset name (e.g.,
            train, valid or test)

        # Inputs
            fname: path to a file.
                *.h5 (HDF5 format) -
                    It will try first read 'feat_name/dt_name' from h5
                    if feat_name doesn't exists will try raw
                    if dt_name doesn't exists will try ''
                *json (JSON format) - a list of dictionary. Must have 'audio'
                and 'label' keys
                    if has key 'dt' will try to only read from specific dt_name

        # Outputs
            If fname is:
                HDF5 format: H5Iterator
                JSON format: JSONIterator
        """
        ext = os.path.splitext(fname)[1]
        dt_iter = None

        if h5py.is_hdf5(fname):
            h5_f = h5py.File(fname, 'r')

            feat_group = h5_f['raw']
            if self.feature_extractor is not None and \
               str(self.feature_extractor) in h5_f.keys():
                feat_group = h5_f[str(self.feature_extractor)]
                self.feature_extractor = None

            if dt_name and dt_name in feat_group.keys():
                dt_iter = self.flow_from_h5(feat_group[dt_name])
            else:
                dt_iter = self.flow_from_h5(feat_group)

            return dt_iter

        if ext == '.json':

            data = {}
            with codecs.open(fname, 'r', encoding='utf8') as json_f:
                ld = json.load(json_f)
                data = utils.ld2dl(ld)

            if dt_name and 'dt' in data and dt_name in set(data['dt']):
                dt_iter = self.flow_from_dl(data, dt_name)
            else:
                dt_iter = self.flow_from_dl(data, None)

            return train_iter

        raise ValueError("Extension not recognized")

    def flows_from_fname(self, fname):
        """ Returns three iterators: train iterator, valid iterator and test iterator

        # Output
            If all dataset name was found: train_iter, valid_iter and test_iter
            Otherwise will try to return only train_iter, None, None
        """
        ext = os.path.splitext(fname)[1]

        train_iter, valid_iter, test_iter = None, None, None

        if h5py.is_hdf5(fname):
            h5_f = h5py.File(fname, 'r')

            feat_group = h5_f['raw']
            if self.feature_extractor is not None and \
               str(self.feature_extractor) in h5_f.keys():
                feat_group = h5_f[str(self.feature_extractor)]
                # it's not necessary, feature already exists
                self.feature_extractor = None

            if 'train' in feat_group.keys():
                train_iter = self.flow_from_h5(feat_group['train'])
            else:
                train_iter = self.flow_from_h5(feat_group)

            if 'valid' in feat_group.keys():
                valid_iter = self.flow_from_h5(feat_group['valid'])

            if 'test' in feat_group.keys():
                test_iter = self.flow_from_h5(feat_group['test'])

            return train_iter, valid_iter, test_iter

        if ext == '.json':

            data = {}
            with codecs.open(fname, 'r', encoding='utf8') as json_f:
                ld = json.load(json_f)
                data = utils.ld2dl(ld)

            if 'dt' in data:
                dts = set(data['dt'])
                if 'train' in dts:
                    train_iter = self.flow_from_dl(data, 'train')

                if 'valid' in dts:
                    valid_iter = self.flow_from_dl(data, 'valid')

                if 'test' in dts:
                    test_iter = self.flow_from_dl(data, 'test')
            else:
                train_iter = self.flow_from_dl(data, None)

            return train_iter, valid_iter, test_iter

        raise ValueError("Extension not recognized")

    def flow_from_json(self, json_fname, dt_sel='train'):
        """ Returns JSONIterator given the filename"""
        return JSONIterator(json_fname, dt_sel, batch_size=self.batch_size,
                            shuffle=self.shuffle, seed=self.seed,
                            feature_extractor=self.feature_extractor,
                            text_parser=self.text_parser)

    def flow_from_dl(self, dl, dt_sel=None):
        """ Return DictListIterator given a list of dictionaries. Each
        dictionary must have the keys 'input' and 'label'
        """
        return DictListIterator(dl, dt_sel, batch_size=self.batch_size,
                                shuffle=self.shuffle, seed=self.seed,
                                feature_extractor=self.feature_extractor,
                                text_parser=self.text_parser)

    def flow_from_h5(self, h5_group=None):
        """ Returns H5Iterator given a h5group from a HDF5 data
        """
        return H5Iterator(h5_group, batch_size=self.batch_size,
                          shuffle=self.shuffle, seed=self.seed,
                          feature_extractor=self.feature_extractor,
                          text_parser=self.text_parser)

    def flow_from_h5_file(self, h5_file, h5_group_name='/'):
        with h5py.File(h5_file, 'r') as f:
            return H5Iterator(f[h5_group_name], batch_size=self.batch_size,
                              shuffle=self.shuffle, seed=self.seed,
                              feature_extractor=self.feature_extractor,
                              text_parser=self.text_parser)

    def flow(self, inputs, labels):
        return DatasetIterator(inputs, labels, batch_size=self.batch_size,
                               shuffle=self.shuffle, seed=self.seed,
                               feature_extractor=self.feature_extractor,
                               text_parser=self.text_parser)


class DatasetIterator(Iterator):

    def __init__(self, inputs, labels, batch_size=32, shuffle=False,
                 seed=None, feature_extractor=None, text_parser=None):
        """ DatasetIterator iterates in a batch over a dataset and do some
        preprocessing on inputs and labels

        # Arguments
            inputs: a list of ndarray
            labels: a list of str or ndarray
            batch_size: size of each batch
            shuffle: if True after each epoch the dataset will shuffle the
            indexes
            seed: seed the random generator
            feature_extractor: instance of Feature
            [preprocessing.audio.Feature]
                feature that is applied to each ndarray in batch
            text_parser: instance of Parser [preprocessing.text.Parser].
                parser that is applied to each label in batch
        """

        if labels is not None and len(inputs) != len(labels):
            raise ValueError('inputs and labels '
                             'should have the same length. '
                             'Found: len(inputs) = %s, len(labels) = %s' %
                             (len(inputs), len(labels)))

        self.inputs = inputs
        self.labels = labels

        self.feature_extractor = feature_extractor
        self.text_parser = text_parser

        super(DatasetIterator, self).__init__(len(inputs), batch_size,
                                              shuffle, seed)

    @property
    def len(self):
        """ Return the total size of dataset
        """
        return len(self.inputs)

    def next(self):
        """ Iterates over batches

        # Outputs
            Returns a tuple (input, output) that can be fed a CTC model
                input: is a list containing the inputs, labels and sequence
                length for the current batch
                output: is a list containing a vector of zeros (fake data for
                the decoder) and the batch labels for the decoder of a CTC
                model
        """

        # Copy from DirectoryIterator from keras
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        index_array.sort()

        batch_inputs, batch_seq_len = self._make_in(self.inputs[index_array.tolist()], current_batch_size)

        batch_labels = self._make_out(self.labels[index_array.tolist()],
                                      current_batch_size)

        return self._make_in_out(batch_inputs, batch_labels, batch_seq_len)

    def _make_in_out(self, batch_inputs, batch_labels, batch_seq_len=None):
        return [batch_inputs, batch_labels, batch_seq_len],
        [np.zeros((batch_inputs.shape[0],)), batch_labels]

    def _make_in(self, inputs, batch_size=None):
        if self.feature_extractor is not None:
            inputs = np.asarray([self.feature_extractor(i) for i in inputs])

        batch_inputs = pad_sequences(inputs, dtype='float32', padding='post')
        batch_seq_len = np.asarray([i.shape[0] for i in inputs])
        return batch_inputs, batch_seq_len

    def _make_out(self, labels, batch_size=None):
        if self.text_parser is not None:
            labels = [self.text_parser(l) for l in labels]

        rows, cols, data = [], [], []

        for row, label in enumerate(labels):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        return scipy.sparse.coo_matrix((data, (rows, cols)), dtype='int32')


class H5Iterator(DatasetIterator):

    def __init__(self, h5group, **kwargs):

        inputs = h5group['inputs']
        labels = h5group['labels']

        if kwargs.get('text_parser') is None:
            raise ValueError("text_parser must be set")

        self.num_feats = None
        if 'num_feats' in inputs.attrs.keys():
            self.num_feats = inputs.attrs['num_feats']

            # Features are computed, no necessity of extract them
            kwargs['feature_extractor'] = None

        self.durations = h5group['durations']

        super(H5Iterator, self).__init__(inputs, labels, **kwargs)

    def _make_in(self, inputs, batch_size=None):
        if self.num_feats is not None:
            inputs = [i.reshape((-1, self.num_feats)) for i in inputs]

        return super(H5Iterator, self)._make_in(inputs)


class JSONIterator(DatasetIterator):

    def __init__(self, json_fname, dt_sel='train', **kwargs):

        kwargs.setdefault('feature_extractor', audio.raw)

        if kwargs.get('feature_extractor') is None:
            raise ValueError("feature_extractor must be set")

        if kwargs.get('text_parser') is None:
            raise ValueError("text_parser must be set")

        with codecs.open(json_fname, 'r', encoding='utf8') as f:
            ld = json.load(f)

        data = utils.ld2dl(ld)

        if 'dt' in data:
            inputs = np.array([d['audio'] for d in data if d['dt'] == dt_sel])
            labels = np.array([d['label'] for d in data if d['dt'] == dt_sel])
        else:
            inputs = np.array(data['audio'])
            labels = np.array(data['label'])

        super(JSONIterator, self).__init__(inputs, labels, **kwargs)

        self.durations = np.array(data['duration'])

class DictListIterator(DatasetIterator):

    def __init__(self, dict_list, dt_sel=None, **kwargs):

        kwargs.setdefault('feature_extractor', audio.raw)

        if kwargs.get('feature_extractor') is None:
            raise ValueError("feature_extractor must be set")

        if kwargs.get('text_parser') is None:
            raise ValueError("text_parser must be set")

        if 'dt' in dict_list and dt_sel is not None:
            dict_list = self._get_by_dt(dict_list, dt_sel)

        inputs = np.array(dict_list['audio'])
        labels = np.array(dict_list['label'])

        super(DictListIterator, self).__init__(inputs, labels, **kwargs)

        self.durations = np.array(dict_list['duration'])

    def _get_by_dt(self, dl, dt):
        mask = [i for i, d in enumerate(dl['dt']) if d == dt]
        return {k: np.array(v)[mask] for k, v in dl.iteritems() if k != 'dt'}
