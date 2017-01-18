import codecs
import json

from datasets import DT_ABSPATH
from datasets.utils import ld2dl

from preprocessing import audio


class DatasetParser(object):
    '''Read data from directory and parse_args
    '''

    def __init__(self, dt_dir):
        self.dt_dir = dt_dir
        self.output_dir = os.path.join(DT_ABSPATH, self.name)
        self.json_fname = os.path.join(self.output_dir, 'data.json')
        self.h5_fname = os.path.join(self.output_dir, 'data.h5')

        self.has_json = False
        if os.path.isfile(self.json_fname):
            self.has_json = True

        self.has_h5 = False
        if os.path.is_file(self.h5_fname):
                self.has_h5 = True

    def _to_ld(self):
        ''' Transform dataset in a list of dictionary
        '''
        data = []
        for d in self._iter(self.dt_dir):
            if not isinstance(d, dict):
                raise TypeError, "__loop must return a dict"

            for k in ['audio', 'label']:
                if not d.has_key(k):
                    raise KeyError, "__loop must return a dict with %s key" % k

            data.append(d)
        return data

    def to_json(self, override=False):

        if self.has_json and overwrite == False:
            raise IOError, "JSON file already exists. If you want to override the current file you must set the parameter `overwrite` to `True`"

        report_fname = os.path.join(self.output_dir, 'json_report.txt')

        data = self._to_ld(self.dt_dir)

        with codecs.open(self.json_fname, 'w', encoding='utf8') as f:
            json.dump(f, data)

        report = self._report(ld2dl(data))
        with open(report_fname, 'w') as f:
            f.write(report + '\n')

    def to_h5(self, feat_map=audio.raw, override=False):
        ''' Generates h5df file for the dataset
        Note that this function will calculate the features rather than store the url to the audio file
        '''

        if not isinstance(feat_map, audio.Feature):
            raise TypeError, "feat_map must be an instance of audio.Feature"

        feat_name = str(feat_map)

        if self.has_h5:
            with h5py.File(self.h5_fname, 'r') as f:
                if feat_name in f.keys() and override == False:
                    raise IOError, "H5 file already exists. If you want to override the current file you must set the parameter `override` to `True`"


        with h5py.File(self.h5_name) as f:

            # If the key already exists
            if feat_name in f.keys():
                del f[feat_name]

            feat_group = f.create_group(feat_name)

            feats = feat_group.create_dataset('inputs', (0,), max_shape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('float32')))
            feats.attr['num_feats'] = num_feats

            labels = feat_group.create_dataset('labels', (0,), max_shape=(None,), dtype=h5py.special_dtype(vlen=unicode))

            total_seq_len, max_labels_len = 0
            for index, data in enumerate(self._to_ld()):

                audio_fname, label = data['audio'], data['label']
                feat = feat_map(audio_fname)

                feats.resize(index + 1)
                feats[index] = feat[:]

                labels.resize(index + 1)
                labels[index] = labels.encode('utf8')

                # Flush to disk only when it reaches 32 samples
                if index % 32:
                    f.flush()

            f.flush()



    def read(self, method=None):
        ''' Read dataset from disk (either json file or from directory) and returns a list of dictionaries

        Args:
            method:
                if `None` will try to read from json and disk (following this order).
                if `json` will try to read from json.
                if `h5` will try to read from h5 file.
                if `dir` will try to read from dataset directory
        '''

        if method not in [None, 'h5', 'json', 'dir']:
            raise ValueError, "method must be one of [None, 'h5', 'json', 'dir']"

        if method == None:
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
        raise NotImplementedError, "_iter must be implemented"

    def _report(self, data):
        raise NotImplementedError, "_report must be implemented"

    def __str__(self):
        raise NotImplementedError, "__str__ must be implemented"
