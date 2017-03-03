import os
import codecs
import json
import h5py

import numpy as np

from datasets import DT_ABSPATH
from common.utils import ld2dl

from preprocessing import audio, text
from common.utils import safe_mkdirs

import logging


class DatasetParser(object):
    '''Read data from directory and parser in a proper format
    '''

    def __init__(self, dt_dir=None, name=None,
                 text_parser=text.CharParser('s|S|a|p'), h5_fname=None):
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        self.dt_dir = dt_dir
        self._name = name
        self.output_dir = os.path.join(DT_ABSPATH, self.name)
        self.json_fname = os.path.join(self.output_dir, 'data.json')
        self.h5_fname = os.path.join(self.output_dir, h5_fname or 'data.h5')
        self.text_parser = text_parser

        self.has_json = False
        if os.path.isfile(self.json_fname):
            self._logger.debug('has_json = True')
            self.has_json = True

        self.has_h5 = False
        if os.path.isfile(self.h5_fname):
                self._logger.debug('has_h5 = True')
                self.has_h5 = True

        if not os.path.isdir(self.output_dir):
            safe_mkdirs(self.output_dir)

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

            for k in ['audio', 'label']:
                if k not in d:
                    raise KeyError("__loop must return a dict with %s key" % k)

            data.append(d)
        return data

    def to_json(self, override=False):

        if self.has_json and override is False:
            raise IOError("JSON file already exists. If you want to override \
                          the current file you must set the parameter \
                          `override` to `True`")

        report_fname = os.path.join(self.output_dir, 'json_report.txt')

        data = self._to_ld()

        with codecs.open(self.json_fname, 'w', encoding='utf8') as f:
            json.dump(data, f)

        report = self._report(ld2dl(data))
        with open(report_fname, 'w') as f:
            logger.info(report)
            f.write(report + '\n')

    def to_h5(self, feat_map=audio.raw, override=False):
        ''' Generates h5df file for the dataset
        Note that this function will calculate the features rather than store
        the url to the audio file
        '''

        if not issubclass(feat_map.__class__, audio.Feature):
            raise TypeError("feat_map must be an instance of audio.Feature")

        feat_name = str(feat_map)

        if self.has_h5:
            with h5py.File(self.h5_fname, 'r') as f:
                if feat_name in f.keys() and override is False:
                    raise IOError("H5 file already exists. If you want to \
                                  override the current file you must set the \
                                  parameter `override` to `True`")

        self._logger.info('Opening %s', self.h5_fname)
        with h5py.File(self.h5_fname) as f:

            # If the key already exists
            if feat_name in f.keys():
                del f[feat_name]

            ld = self._to_ld()

            # handle with multiple datasets
            def create_datasets(feat_group):
                feats = feat_group.create_dataset(
                    'inputs', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=np.dtype('float32')))

                if feat_map.num_feats:
                    feats.attrs['num_feats'] = feat_map.num_feats

                labels = feat_group.create_dataset(
                    'labels', (0,), maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=unicode))

                durations = feat_group.create_dataset(
                    'durations', (0,), maxshape=(None,))

            self._logger.debug('Creating %s group in hdf5 file', feat_name)
            feat_group = f.create_group(feat_name)

            if 'dt' in ld[0]:
                for t in set([d['dt'] for d in ld]):
                    feat_group.create_group(t)
                    create_datasets(feat_group[t])

                get_groups = lambda x: [feat_group['%s/%s' % (x['dt'], dt)]
                                        for dt in ('inputs', 'labels',
                                                   'durations')]
            else:
                get_groups = lambda x: [feat_group['%s' % (dt)] for dt in
                                        ('inputs', 'labels', 'durations')]
                create_datasets(feat_group)

            for index, data in enumerate(ld):

                feats, labels, durations = get_groups(data)

                audio_fname, label, duration = data['audio'], data['label'], data['duration']
                feat = feat_map(audio_fname)

                feats.resize(feats.shape[0] + 1, axis=0)
                feats[feats.shape[0] - 1] = feat.flatten().astype('float32')

                labels.resize(labels.shape[0] + 1, axis=0)
                labels[labels.shape[0] - 1] = label.encode('utf8')

                durations.resize(durations.shape[0] + 1, axis=0)
                durations[durations.shape[0] - 1] = duration

                # Flush to disk only when it reaches 128 samples
                if index % 128 == 0:
                    self._logger.info('%d/%d done.' % (index, len(ld)))
                    f.flush()

            f.flush()
            self._logger.info('%d/%d done.' % (len(ld), len(ld)))

    def read(self, method=None):
        ''' Read dataset from disk (either json file or from directory) and
        returns a list of dictionaries

        Args:
            method:
                if `None` will try to read from json and disk (following this
                order).
                if `json` will try to read from json.
                if `h5` will try to read from h5 file.
                if `dir` will try to read from dataset directory
        '''

        if method not in [None, 'h5', 'json', 'dir']:
            raise ValueError("method must be one of [None, 'h5', 'json', \
                             'dir']")

        if method is None:
            if self.has_json:
                return self._read_from_json()
            else:
                return self._read_from_dir()
        elif method == 'json':
            return self._read_from_json()
        elif method == 'h5':
            return self._read_from_h5()
        elif method == 'dir':
            return self._read_from_dir()

    def _read_from_json(self):
        return json.load(codecs.open(self.json_fname, 'r', encoding='utf8'))

    def _read_from_h5(self):
        return h5py.File(self.h5_fname, 'r')

    def _read_from_dir(self):
        return self._to_ld()

    def _iter(self):
        raise NotImplementedError("_iter must be implemented")

    def _report(self, dl):
        raise NotImplementedError("_report must be implemented")

    def _is_valid_label(self, label):
        if len(label) == 0:
            return False

        if self.text_parser is None:
            return True

        return self.text_parser.is_valid(label)

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self.name
