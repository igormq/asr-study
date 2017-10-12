from __future__ import absolute_import, division, print_function

import os
import codecs
import json

import logging
import h5py

import numpy as np

from preprocessing import audio, text
from datasets import DT_ABSPATH
from utils.generic_utils import safe_mkdirs, ld2dl


class DatasetParser(object):
    '''Read data from directory and parser in a proper format
    '''

    def __init__(self, dataset_dir, name=None):
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        self.dataset_dir = dataset_dir
        self._name = name

        self.default_output_dir = os.path.join(DT_ABSPATH, self.name)

    @property
    def dataset_dir(self):
        """Filepath to the dataset directory"""
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, value):
        if value is None:
            raise ValueError("You must set the variable dataset_dir (the location of dataset) before continue")

        if not os.path.isdir(value):
            raise ValueError("Dataset directory provided is not a directory")
        self._dataset_dir = value

    def _to_ld(self, label_parser=None):
        ''' Transform dataset in a list of dictionary
        '''
        data = []
        for d in self._iter():
            if not isinstance(d, dict):
                raise TypeError("__loop must return a dict")

            for k in ['input', 'label', 'duration']:
                if k not in d:
                    raise KeyError("__loop must return a dict with %s key" % k)

            if not self._is_valid_label(d['label'], label_parser=label_parser):
                self._logger.warning(u'File %s has a forbidden label: "%s". Skipping', d['input'], d['label'])
                continue

            data.append(d)
        return data

    def to_json(self, fname=None):
        ''' Parse the entire dataset to a list of dictionary containin at least
        two keys:
            `input`: path to audio file
            `duration`: length of the audio
            `label`: transcription of the audio
        '''
        fname = fname or os.path.join(
            self.default_output_dir, 'data.json')

        if os.path.exists(fname) and override:
            os.remove(fname)

        if not os.path.isdir(os.path.split(fname)[0]):
            safe_mkdirs(os.path.split(fname)[0])

        data = self._to_ld()

        with codecs.open(fname, 'w', encoding='utf8') as f:
            json.dump(data, f)

        self._logger.info(self._report(ld2dl(data)))

        return fname

    def to_h5(self, fname=None, input_parser=audio.raw, label_parser=None,
              split_sets=True, override=False):
        ''' Generates h5df file for the dataset
        Note that this function will calculate the features rather than store
        the path to the audio file

        Args
            split_sets: if True and dataset is split in several sets (e.g.
            train, valid, test) the h5 file will create the corresponding
            datasets; otherwise no dataset is create
        '''
        if not issubclass(input_parser.__class__, audio.Feature):
            raise TypeError("input_parser must be an instance of audio.Feature")

        fname = fname or os.path.join(self.default_output_dir, 'data.h5')

        if h5py.is_hdf5(fname) and override:
            os.remove(fname)

        if not os.path.isdir(os.path.split(fname)[0]):
            safe_mkdirs(os.path.split(fname)[0])

        feat_name = str(input_parser)

        data = self._to_ld(label_parser=label_parser)

        if len(data) == 0:
            raise IndexError("Data is empty")

        datasets = ['/']
        if 'dataset' in data[0]:
            datasets = list(set([d['dataset'] for d in data]))

        self._logger.info('Opening %s', fname)
        with h5py.File(fname) as f:

            # create all datasets
            for dataset in datasets:

                group = f['/']
                if dataset != '/':
                    group = f.create_group(dataset)

                inputs = group.create_dataset(
                    'inputs', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=np.dtype('float32')))

                if input_parser.num_feats:
                    inputs.attrs['num_feats'] = input_parser.num_feats

                group.create_dataset(
                    'labels', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=unicode))

                group.create_dataset(
                    'durations', (0,), maxshape=(None,))

            for i, d in enumerate(data):

                dataset = '/'
                if dataset not in datasets:
                    dataset = d['dataset']

                # HDF5 pointers
                inputs = f[dataset]['inputs']
                labels = f[dataset]['labels']
                durations = f[dataset]['durations']

                # Data
                input_ = input_parser(d['input'])
                label = d['label']
                duration = d['duration']

                inputs.resize(inputs.shape[0] + 1, axis=0)
                inputs[inputs.shape[0] - 1] = input_.flatten().astype('float32')

                labels.resize(labels.shape[0] + 1, axis=0)
                labels[labels.shape[0] - 1] = label.encode('utf8')

                durations.resize(durations.shape[0] + 1, axis=0)
                durations[durations.shape[0] - 1] = duration

                # Flush to disk only when it reaches 128 samples
                if i % 128 == 0:
                    self._logger.info('%d/%d done.' % (i, len(data)))
                    f.flush()

            f.flush()
            self._logger.info('%d/%d done.' % (len(data), len(data)))

            return fname

    def _iter(self):
        raise NotImplementedError("_iter must be implemented")

    def _report(self, dl):
        """
        Args
            dl: dictionary of list, where the keys were defined in _iter()
        """
        raise NotImplementedError("_report must be implemented")

    def _is_valid_label(self, label, label_parser=None):
        if len(label) == 0:
            return False

        if label_parser is not None:
            return label_parser.is_valid(label)

        return True

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self.name
