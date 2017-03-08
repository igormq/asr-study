from __future__ import absolute_import, division, print_function

import os
import codecs
import json
import h5py

import numpy as np

from preprocessing import audio, text
from datasets import DT_ABSPATH
from common.utils import safe_mkdirs
from common.utils import ld2dl

import logging


class DatasetParser(object):
    '''Read data from directory and parser in a proper format
    '''

    def __init__(self, dt_dir=None, name=None):
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        self.dt_dir = dt_dir
        self._name = name

        self.default_output_dir = os.path.join(DT_ABSPATH, self.name)

    @property
    def dt_dir(self):
        """Filepath to the dataset directory"""
        if self._dt_dir is None:
            raise ValueError("You must set the variable dt_dir (the location \
                             of dataset) before continue")
        return self._dt_dir

    @dt_dir.setter
    def dt_dir(self, value):
        self._dt_dir = value

    def _to_ld(self):
        ''' Transform dataset in a list of dictionary
        '''
        data = []
        for d in self._iter():
            if not isinstance(d, dict):
                raise TypeError("__loop must return a dict")

            for k in ['input', 'label', 'duration']:
                if k not in d:
                    raise KeyError("__loop must return a dict with %s key" % k)

            data.append(d)
        return data

    def to_json(self, json_fname=None):
        ''' Parse the entire dataset to a list of dictionary containin at least
        two keys:
            `input`: path to audio file
            `duration`: length of the audio
            `label`: transcription of the audio
        '''
        json_fname = json_fname or os.path.join(
            self.default_output_dir, 'data.json')

        if os.path.exists(json_fname) and override:
            os.remove(json_fname)

        if not os.path.isdir(os.path.split(json_fname)[0]):
            safe_mkdirs(os.path.split(json_fname)[0])

        data = self._to_ld()

        with codecs.open(json_fname, 'w', encoding='utf8') as f:
            json.dump(data, f)

        print(self._report(ld2dl(data)))

    def to_h5(self, h5_fname=None, feat_map=audio.raw,
              split_sets=True, override=False):
        ''' Generates h5df file for the dataset
        Note that this function will calculate the features rather than store
        the path to the audio file

        Args
            split_sets: if True and dataset is split in several sets (e.g.
            train, valid, test) the h5 file will create the corresponding
            datasets; otherwise no dataset is create
        '''
        if not issubclass(feat_map.__class__, audio.Feature):
            raise TypeError("feat_map must be an instance of audio.Feature")

        h5_fname = h5_fname or os.path.join(self.default_output_dir, 'data.h5')

        if h5py.is_hdf5(h5_fname) and override:
            os.remove(h5_fname)

        if not os.path.isdir(os.path.split(h5_fname)[0]):
            safe_mkdirs(os.path.split(h5_fname)[0])

        feat_name = str(feat_map)

        data = self._to_ld()
        datasets = ['/']
        if 'dataset' in data[0]:
            datasets = list(set([d['dataset'] for d in data]))

        self._logger.info('Opening %s', h5_fname)
        with h5py.File(h5_fname) as f:

            # create all datasets
            for dataset in datasets:

                group = f['/']
                if dataset != '/':
                    group = f.create_group(dataset)

                inputs = group.create_dataset(
                    'inputs', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=np.dtype('float32')))

                if feat_map.num_feats:
                    inputs.attrs['num_feats'] = feat_map.num_feats

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
                input_ = feat_map(d['input'])
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

    def _iter(self):
        raise NotImplementedError("_iter must be implemented")

    def _report(self, dl):
        raise NotImplementedError("_report must be implemented")

    def _is_valid_label(self, label):
        if len(label) == 0:
            return False

        return True

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self.name
