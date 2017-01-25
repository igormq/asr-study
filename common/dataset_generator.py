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
    ''' Generates mini-batches from json file
    '''

    def __init__(self, feature_extractor=None,
                       text_parser=None):
        self.feature_extractor = feature_extractor
        self.text_parser = text_parser

    def flow_from_fname(self, fname, dt_name=None, batch_size=32, shuffle=True, seed=None):
        ext = os.path.splitext(fname)[1]
        dt_iter = None

        if h5py.is_hdf5(fname):
            h5_f =  h5py.File(fname, 'r')

            feat_group = h5_f['raw']
            if self.feature_extractor is not None and str(self.feature_extractor) in h5_f.keys():
                feat_group = h5_f[str(self.feature_extractor)]

            if dt_name and dt_name in feat_group.keys():
                dt_iter = self.flow_from_h5(feat_group[dt_name], batch_size, shuffle, seed)
            else:
                dt_iter = self.flow_from_h5(feat_group, batch_size, shuffle, seed)

            return dt_iter

        if ext == '.json':

            data = {}
            with codecs.open(fname, 'r', encoding='utf8') as json_f:
                ld = json.load(json_f)
                data = utils.ld2dl(ld)

            if dt_name and data.has_key('dt') and dt_name in set(data['dt']):
                dt_iter = self.flow_from_dl(data, dt_name, batch_size, shuffle, seed)
            else:
                dt_iter = self.flow_from_dl(data, None, batch_size, shuffle, seed)

            return train_iter

        raise ValueError, "Extension not recognized"

    def flows_from_fname(self, fname, batch_size=32, shuffle=True, seed=None, split=[1., 0., 0.]):
        ext = os.path.splitext(fname)[1]

        train_iter, valid_iter, test_iter = None, None, None

        if h5py.is_hdf5(fname):
            h5_f =  h5py.File(fname, 'r')

            feat_group = h5_f['raw']
            if self.feature_extractor is not None and str(self.feature_extractor) in h5_f.keys():
                feat_group = h5_f[str(self.feature_extractor)]

            if 'train' in feat_group.keys():
                train_iter = self.flow_from_h5(feat_group['train'], batch_size, shuffle, seed)
            else:
                train_iter = self.flow_from_h5(feat_group, batch_size, shuffle, seed)

            if 'valid' in feat_group.keys():
                valid_iter = self.flow_from_h5(feat_group['valid'], batch_size, shuffle, seed)

            if 'test' in feat_group.keys():
                test_iter = self.flow_from_h5(feat_group['test'], batch_size, shuffle, seed)

            return train_iter, valid_iter, test_iter

        if ext == '.json':

            data = {}
            with codecs.open(fname, 'r', encoding='utf8') as json_f:
                ld = json.load(json_f)
                data = utils.ld2dl(ld)

            if data.has_key('dt'):
                dts = set(data['dt'])
                if 'train' in dts:
                    train_iter = self.flow_from_dl(data, 'train', batch_size, shuffle, seed)

                if 'valid' in dts:
                    valid_iter = self.flow_from_dl(data, 'valid', batch_size, shuffle, seed)

                if 'test' in dts:
                    test_iter = self.flow_from_dl(data, 'test', batch_size, shuffle, seed)
            else:
                train_iter = self.flow_from_dl(data, None, batch_size, shuffle, seed)

            return train_iter, valid_iter, test_iter

        raise ValueError, "Extension not recognized"

    def flow_from_json(self, json_fname, dt_sel='train', batch_size=32, shuffle=True, seed=None):
        return JSONIterator(json_fname, dt_sel, batch_size, shuffle, seed, self.feature_extractor, self.text_parser)

    def flow_from_dl(self, dl, dt_sel=None, batch_size=32, shuffle=True, seed=None):
        return DictListIterator(dl, dt_sel, batch_size, shuffle, seed, self.feature_extractor, self.text_parser)

    def flow_from_h5(self, h5_group=None, batch_size=32, shuffle=True, seed=None):
        return H5Iterator(h5_group, batch_size, shuffle, seed, self.feature_extractor, self.text_parser)

    def flow_from_h5_file(self, h5_file, h5_group_name='/', batch_size=32, shuffle=True, seed=None):
        with h5py.File(h5_file, 'r') as f:
            return H5Iterator(f[h5_group_name], batch_size, shuffle, seed, self.feature_extractor, self.text_parser)

    def flow(self, inputs, labels, batch_size=32, shuffle=True, seed=None):
        return DatasetIterator(inputs, labels, batch_size, shuffle, seed, self.feature_extractor, self.text_parser)

class DatasetIterator(Iterator):

    def __init__(self, inputs, labels, batch_size=32, shuffle=False, seed=None, feature_extractor=None, text_parser=None):

        if labels is not None and len(inputs) != len(labels):
            raise ValueError('inputs and labels '
                             'should have the same length. '
                             'Found: len(inputs) = %s, len(labels) = %s' % (len(inputs), len(labels)))

        self.inputs = inputs
        self.labels = labels

        self.feature_extractor = feature_extractor
        self.text_parser = text_parser

        super(DatasetIterator, self).__init__(len(inputs), batch_size, shuffle, seed)

    @property
    def len(self):
        return len(self.inputs)


    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        index_array.sort()

        batch_inputs, batch_seq_len = self._make_in(self.inputs[index_array.tolist()], current_batch_size)
        batch_labels = self._make_out(self.labels[index_array.tolist()], current_batch_size)

        return self._make_in_out(batch_inputs, batch_labels, batch_seq_len)

    def _make_in_out(self, batch_inputs, batch_labels, batch_seq_len=None):
        return [batch_inputs, batch_labels, batch_seq_len], [np.zeros((batch_inputs.shape[0],)), batch_labels]

    def _make_in(self, inputs, batch_size=None):
        if self.feature_extractor is not None:
            inputs = np.asarray([self.feature_extractor(i) for i in inputs])

        batch_inputs = pad_sequences(inputs, dtype='float32', padding='post')
        batch_seq_len = np.asarray([i.shape[0] for i in inputs])
        return batch_inputs, batch_seq_len

    def _make_out(self, labels, batch_size=None):
        if self.text_parser is not None:
            labels = [self.text_parser(l) for l in labels]

        max_labels_size = np.max([len(l) for l in labels])
        batch_labels = scipy.sparse.lil_matrix((batch_size, max_labels_size), dtype='int32')
        for i, l in enumerate(labels):
             batch_labels[i, :l.size] = l
        return batch_labels

class H5Iterator(DatasetIterator):

    def __init__(self, h5group, batch_size=32, shuffle=False, seed=None, feature_extractor=None, text_parser=None):

        inputs = h5group['inputs']
        labels = h5group['labels']

        if text_parser is None:
            raise ValueError, "text_parser must be set"

        super(H5Iterator, self).__init__(inputs, labels, batch_size, shuffle, seed, feature_extractor, text_parser)

        self.num_feats = None
        if 'num_feats' in inputs.attrs.keys():
            self.num_feats = inputs.attrs['num_feats']

        self.durations = h5group['durations']

    def _make_in(self, inputs, batch_size=None):
        if self.num_feats is not None:
            inputs = [i.reshape((-1, self.num_feats)) for i in inputs]

        return super(H5Iterator, self)._make_in(inputs)

class JSONIterator(DatasetIterator):

    def __init__(self, json_fname, dt_sel='train', batch_size=32, shuffle=False, seed=None, feature_extractor=audio.raw, text_parser=None):

        if feature_extractor is None:
            raise ValueError, "feature_extractor must be set"

        if text_parser is None:
            raise ValueError, "text_parser must be set"

        with codecs.open(json_fname, 'r', encoding='utf8') as f:
            ld = json.load(f)

        data = utils.ld2dl(ld)

        if data.has_key('dt'):
            inputs = np.array([d['audio'] for d in data if d['dt'] == dt_sel])
            labels = np.array([d['label'] for d in data if d['dt'] == dt_sel])
        else:
            inputs = np.array(data['audio'])
            labels = np.array(data['label'])

        super(JSONIterator, self).__init__(inputs, labels, batch_size, shuffle, seed, feature_extractor, text_parser)

        self.durations = np.array(data['duration'])

class DictListIterator(DatasetIterator):

    def __init__(self, dict_list, dt_sel=None, batch_size=32, shuffle=False, seed=None, feature_extractor=audio.raw, text_parser=None):

        if feature_extractor is None:
            raise ValueError, "feature_extractor must be set"

        if text_parser is None:
            raise ValueError, "text_parser must be set"

        if dict_list.has_key('dt') and dt_sel is not None:
            dict_list = self._get_by_dt(dict_list, dt_sel)

        inputs = np.array(dict_list['audio'])
        labels = np.array(dict_list['label'])

        super(DictListIterator, self).__init__(inputs, labels, batch_size, shuffle, seed, feature_extractor, text_parser)

        self.durations = np.array(dict_list['duration'])


    def _get_by_dt(self, dl, dt):
        mask = [i for i, d in enumerate(dl['dt']) if d == dt]
        return {k: np.array(v)[mask] for k,v in dl.iteritems() if k != 'dt'}
