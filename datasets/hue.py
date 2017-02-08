from datasets import DatasetParser
from datasets import CSLU, VoxForge, Sidney, LapsBM

import os
import re
import librosa
import codecs


class HUE(DatasetParser):
    """ HUE dataset reader and parser

    This dataset is a combination of four smaller datasets (voxforge, lapsbm,
    sidney, and cslu spoltech port). The dataset was divided in the following
    way:
        * Train: voxforge, sidney, and cslu spoltech port
        * Valid: 5 women and 15 men from LaspBM
        * Test: 5 women 10 men from LapsBM (without overlapping with valid set
        either in speaker and utterance spoken)

    After cleaning (removing label with zero length, label with numeric
    digits, e.g., 4 instead of four) the training set contains 11702
    utterances with 425 speakers.

    """

    def __init__(self, **kwargs):

        kwargs.setdefault('name', 'hue')

        super(HUE, self).__init__(**kwargs)

    def dt_dir(self):
        """Filepath to the dataset directory"""
        value = super(dt_dir, self).dt_dir

        if not isinstance(value, dict):
            raise ValueError("dt_dir must be a dictionary")

        for key in ('cslu', 'lapsbm', 'voxforge', 'sidney'):
            if key not in value:
                raise KeyError("dt_dir must have the key %s" % key)

        return self._dt_dir

    def _iter(self):

        for dataset in (CSLU(dt_dir=self.dt_dir['cslu']),
                        VoxForge(dt_dir=self.dt_dir['voxforge']),
                        Sidney(dt_dir=self.dt_dir['sidney'])):

            for d in dataset._iter():
                yield {'duration': d['duration'],
                       'audio': d['audio'],
                       'label': d['label'],
                       'speaker': '%s_%s' % (str(dataset), d['speaker']),
                       'dt': 'train'}

        # Test and valid set
        lapsbm = LapsBM(dt_dir=self.dt_dir['lapsbm'], split=True)
        for d in lapsbm._iter():
            yield {'duration': d['duration'],
                   'audio': d['audio'],
                   'label': d['label'],
                   'speaker': '%s_%s' % (str(dataset), d['speaker']),
                   'dt': d['dt']}

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           Number of speakers: %d''' % (len(dl['audio']),
                                        sum(dl['duration']),
                                        len(set(dl['speaker'])))

        return report
