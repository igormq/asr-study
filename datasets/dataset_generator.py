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

import time

from preprocessing import audio, text
from utils import generic_utils as utils

import logging


class DatasetGenerator(object):
    """ Dataset generator that handles several forms of input and return an
    iterator over it. Only works for a CTC model

    # Arguments
        input_parser: instance of Feature [preprocessing.audio.Feature]
            feature that is applied to each audio file (or audio data)
        label_parser: instance of Parser [preprocessing.text.Parser].
            parser that is applied to each label data
        batch_size: number of samples per batch
        shuffle: reordering index per epoch. This avoid some bias in training
        seed: default None
    """

    def __init__(self, input_parser=None, label_parser=None, batch_size=32,
                 shuffle=True, seed=None, mode='train'):
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        self.input_parser = input_parser
        self.label_parser = label_parser
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mode = mode

    def flow_from_fname(self, fname, datasets=None):
        """ Returns an specific iterator given the filename

        # Arguments
            datasets: str or list. If str will return one iterator; otherwise
            will return len(dataset) iterators for each dataset

        # Inputs
            fname: path to a file.
                *.h5 (HDF5 format)
                *json (JSON format)

        # Outputs
            If fname is:
                HDF5 format: H5Iterator
                JSON format: JSONIterator
        """
        out = None
        datasets = datasets or ['/']
        if type(datasets) not in (set, list):
            datasets = [datasets]

        if h5py.is_hdf5(fname):
            h5_f = h5py.File(fname, 'r')
            out = [self.flow_from_h5_group(h5_f[dataset])
                   for dataset in datasets]

        ext = os.path.splitext(fname)[1]
        if ext == '.json':
            out = [self.flow_from_json(fname, dataset) for dataset in datasets]

        if out is None:
            raise ValueError("Extension not recognized")

        if len(out) == 1:
            return out[0]
        return out

    def flow_from_json(self, fname, dataset=None):
        """ Returns JSONIterator given the filename"""
        return JSONIterator(
            fname, dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, seed=self.seed,
            input_parser=self.input_parser,
            label_parser=self.label_parser,
            mode=self.mode)

    def flow_from_dl(self, dl, dataset=None):
        """ Return DictListIterator given a list of dictionaries. Each
        dictionary must have the keys 'input' and 'label'
        """
        return DictListIterator(dl, dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, seed=self.seed,
                                input_parser=self.input_parser,
                                label_parser=self.label_parser,
                                mode=self.mode)

    def flow_from_h5_group(self, h5_group=None):
        """ Returns H5Iterator given a h5group from a HDF5 data
        """
        return H5Iterator(h5_group, batch_size=self.batch_size,
                          shuffle=self.shuffle, seed=self.seed,
                          input_parser=self.input_parser,
                          label_parser=self.label_parser,
                          mode=self.mode)

    def flow_from_h5_file(self, h5_file, dataset='/'):
        h5_f = h5py.File(h5_file, 'r')
        return H5Iterator(h5_f[dataset], batch_size=self.batch_size,
                          shuffle=self.shuffle, seed=self.seed,
                          input_parser=self.input_parser,
                          label_parser=self.label_parser,
                          mode=self.mode)

    def flow(self, inputs, labels):
        return DatasetIterator(inputs, labels, batch_size=self.batch_size,
                               shuffle=self.shuffle, seed=self.seed,
                               input_parser=self.input_parser,
                               label_parser=self.label_parser,
                               mode=self.mode)


class DatasetIterator(Iterator):

    def __init__(self, inputs, labels=None, batch_size=32, shuffle=False,
                 seed=None, input_parser=None, label_parser=None,
                 standarize=None, mode='train'):
        """ DatasetIterator iterates in a batch over a dataset and do some
        preprocessing on inputs and labels

        # Arguments
            inputs: a list of ndarray
            labels: a list of str or ndarray
            batch_size: size of each batch
            shuffle: if True after each epoch the dataset will shuffle the
            indexes
            seed: seed the random generator
            input_parser: instance of Feature
            [preprocessing.audio.Feature]
                feature that is applied to each ndarray in batch
            label_parser: instance of Parser [preprocessing.text.Parser].
                parser that is applied to each label in batch
            standarize: if a set (mean, std), the input will be
            normalized
            mode: if 'predict', only the inputs is generated
        """

        if labels is not None and len(inputs) != len(labels):
            raise ValueError('inputs and labels '
                             'should have the same length. '
                             'Found: len(inputs) = %s, len(labels) = %s' %
                             (len(inputs), len(labels)))
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        self.inputs = inputs
        self.labels = labels

        self.input_parser = input_parser
        self.label_parser = label_parser

        self.standarize = standarize
        self.mode = mode

        if self.input_parser is not None:
            logging.warning('Feature extractor is not None. It may slow down'
                            + ' training')

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
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        index_array.sort()

        index_array_list = index_array.tolist()

        batch_inputs, batch_inputs_len = self._make_in(
            self.inputs[index_array_list], current_batch_size)

        if self.labels is not None:
            batch_labels = self._make_out(self.labels[index_array_list],
                                          current_batch_size)
        else:
            batch_labels = None

        return self._make_in_out(batch_inputs, batch_labels, batch_inputs_len)

    def _make_in_out(self, batch_inputs, batch_labels, batch_inputs_len=None):
        # if label is not provided output is not necessary
        if batch_labels is None:
            return [batch_inputs, batch_inputs_len]

        return ([batch_inputs, batch_labels, batch_inputs_len],
                [np.zeros((batch_inputs.shape[0],)), batch_labels])

    def _make_in(self, inputs, batch_size=None):
        if self.input_parser is not None:
            inputs = np.asarray([self.input_parser(i) for i in inputs])

        batch_inputs = pad_sequences(inputs, dtype='float32', padding='post')

        if self.standarize:
            mean, std = self.standarize
            batch_inputs -= mean
            batch_inputs /= (std + self.eps)

        batch_inputs_len = np.asarray([i.shape[0] for i in inputs])
        return batch_inputs, batch_inputs_len

    def _make_out(self, labels, batch_size=None):
        if self.labels is None or self.mode == 'predict':
            return None

        if self.label_parser is not None:
            labels = [self.label_parser(l) for l in labels]

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

        if kwargs.get('label_parser') is None:
            raise ValueError("label_parser must be set")

        self.num_feats = None
        if 'num_feats' in inputs.attrs.keys():
            self.num_feats = inputs.attrs['num_feats']

        self.durations = h5group['durations']

        super(H5Iterator, self).__init__(inputs, labels, **kwargs)

    def _make_in(self, inputs, batch_size=None):

        if self.num_feats is not None:
            inputs = [i.reshape((-1, self.num_feats)) for i in inputs]

        return super(H5Iterator, self)._make_in(inputs)


class JSONIterator(DatasetIterator):

    def __init__(self, fname, dataset=None, **kwargs):

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

        kwargs.setdefault('input_parser', audio.raw)

        if kwargs.get('input_parser') is None:
            raise ValueError("input_parser must be set")

        if kwargs.get('label_parser') is None:
            raise ValueError("label_parser must be set")

        with codecs.open(fname, 'r', encoding='utf8') as f:
            ld = json.load(f)

        data = utils.ld2dl(ld)

        if dataset and 'dataset' not in data:
            self._logger.warning('No dataset key found. Falling back to None')
            dataset = None

        if dataset:
            inputs = np.array([i for i, d in zip(
                data['input'], data['dataset']) if d == dataset])
            labels = np.array([l for l, d in zip(
                data['label'], data['dataset']) if d == dataset])
        else:
            inputs = np.array(data['input'])
            labels = np.array(data['label'])

        super(JSONIterator, self).__init__(inputs, labels, **kwargs)

        self.durations = np.array(data['duration'])


class DictListIterator(DatasetIterator):

    def __init__(self, dict_list, dataset=None, **kwargs):

        kwargs.setdefault('input_parser', audio.raw)

        if kwargs.get('input_parser') is None:
            raise ValueError("input_parser must be set")

        if kwargs.get('label_parser') is None:
            raise ValueError("label_parser must be set")

        if dataset:
            dict_list = self._get_by_dataset(dict_list, dataset)

        inputs = np.array(dict_list['audio'])
        labels = np.array(dict_list['label'])

        super(DictListIterator, self).__init__(inputs, labels, **kwargs)

        self.durations = np.array(dict_list['duration'])

    def _get_by_dataset(self, dl, dataset):
        mask = [i for i, d in enumerate(dl['dataset']) if d == dataset]
        return {k: np.array(v)[mask] for k, v in dl.iteritems()
                if k != 'dataset'}
