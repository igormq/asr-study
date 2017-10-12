from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import DatasetParser
from datasets import LapsBM

from utils.generic_utils import get_from_module


class BRSD(DatasetParser):
    """ Brazilian Portuguese Speech dataset reader and parser

    This dataset is a combination of four smaller datasets (voxforge, lapsbm,
    sid, and cslu spoltech port). The dataset was divided in the following
    way:
        * Train: voxforge, sid, and cslu spoltech port
        * Valid: 5 women and 15 men from LaspBM
        * Test: 5 women 10 men from LapsBM (without overlapping with valid set
        either in speaker and utterance spoken)

    After cleaning (removing label with zero length, label with numeric
    digits, e.g., 4 instead of four) the training set contains 11702
    utterances with 425 speakers.

    """

    def __init__(self, dataset_dir=None, name='brsd', **kwargs):

        dataset_dir = dataset_dir or {'lapsbm': None,
                                      'voxforge': None,
                                      'sid': None,
                                      'cslu': None}

        super(BRSD, self).__init__(dataset_dir, name, **kwargs)

    @property
    def dataset_dir(self):
        """Filepath to the dataset directory"""
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, value):
        """Filepath to the dataset directory"""

        if value is None:
            raise ValueError("You must set the variable dataset_dir"
                             " (the location of dataset) before continue")

        if not isinstance(value, dict):
            raise ValueError("dataset_dir must be a dictionary")

        for key in ('lapsbm', 'voxforge', 'sid'):
            if key not in value:
                raise KeyError("dataset_dir must have the key %s" % key)

        if 'cslu' not in value:
            self._logger.warning('CSLU not found. Ignoring it.')

        self._dataset_dir = value

    def _iter(self):

        for name, path in self.dataset_dir.items():

            if name == 'lapsbm':
                continue

            try:
                dataset_cls = get_from_module('datasets*', name, regex=True)
                dataset = dataset_cls(dataset_dir=path)

                for d in dataset._iter():
                    yield {'duration': d['duration'],
                           'input': d['input'],
                           'label': d['label'],
                           'speaker': '%s_%s' % (str(dataset), d['speaker']),
                           'dataset': 'train'}
            except ValueError, e:
                self._logger.warning('Skipping dataset %s: %s' % (name, e.message))
        # Test and valid set
        lapsbm = LapsBM(dataset_dir=self.dataset_dir['lapsbm'], split=True)
        for d in lapsbm._iter():
            yield {'duration': d['duration'],
                   'input': d['input'],
                   'label': d['label'],
                   'speaker': '%s_%s' % (str(dataset), d['speaker']),
                   'dataset': d['dataset']}

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           Number of speakers: %d''' % (len(dl['input']),
                                        sum(dl['duration']),
                                        len(set(dl['speaker'])))

        return report
