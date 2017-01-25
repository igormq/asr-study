from datasets import DatasetParser
from datasets import CSLU, VoxForge, Sidney, LapsBM

import os
import re
import librosa
import codecs

class HUE(DatasetParser):

    def dt_dir(self):
        """Filepath to the dataset directory"""
        value = super(dt_dir, self).dt_dir

        if not isinstance(value, dict):
            raise ValueError, "dt_dir must be a dictionary"

        for key in ('cslu', 'lapsbm', 'voxforge', 'sidney'):
            if not value.has_key(key):
                raise KeyError, "dt_dir must have the key %s" % key

        return self._dt_dir

    def _iter(self):

        for dataset in (CSLU(self.dt_dir['cslu']),
                        VoxForge(self.dt_dir['voxforge']),
                        Sidney(self.dt_dir['sidney'])):

            for d in dataset._iter():
                yield {'duration': d['duration'],
                       'audio': d['audio'],
                       'label': d['label'],
                       'speaker': '%s_%s' % (str(dataset), d['speaker']),
                       'dt': 'train'}
       
        # Test and valid set
        lapsbm = LapsBM(self.dt_dir['lapsbm'], split=True)
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
           Number of speakers: %d''' % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])))

        return report

    def __str__(self):
        return 'hue'
