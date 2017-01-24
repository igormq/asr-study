from datasets import DatasetParser

import os
import re
import librosa
import codecs
import tempfile

import numpy as np

class Dummy(DatasetParser):

    def __init__(self, nb_speakers=10, nb_utterances_per_speaker=10, max_duration=10.0, min_duration=1.0, max_label_length=200, fs=16e3, name='dummy', split=None):
        '''
        Args:
            split: list or nparray of size 2 that splits the data between train, valid and test. example: split = [.8 .15] = 80% train, 15% valid and 5% test
        '''
        self._name = name

        super(Dummy, self).__init__(None)

        self.nb_speakers = nb_speakers
        self.nb_utterances_per_speaker = nb_utterances_per_speaker
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.fs = fs
        self.max_label_length = max_label_length
        self.split = split

        if split is not None and (len(split) != 2 or np.sum(split) > 1.):
            raise ValueError('Split must have len = 2 and must sum <= 1')

    def _iter(self):

        counter = 0
        total = self.nb_speakers * self.nb_utterances_per_speaker

        for speaker in range(self.nb_speakers):
            for utterance in range(self.nb_utterances_per_speaker):

                duration = np.random.uniform(low=self.min_duration, high=self.max_duration)

                samples = np.floor(duration*self.fs)
                audio = np.random.randn(int(samples))

                audio_file = tempfile.NamedTemporaryFile(delete=False)
                audio_fname = audio_file.name
                audio_file.close()

                librosa.output.write_wav(audio_fname, audio, self.fs)

                label = np.random.randint(low=ord('a'), high=ord('z'), size=(np.random.randint(2, self.max_label_length),))
                label = ''.join([chr(l) for l in label])

                data = {'duration': duration,
                       'audio': audio_fname,
                       'label': label,
                       'speaker': 'speaker_%d' % speaker}

                if self.split is not None:
                    if counter < np.floor(self.split[0]*total):
                        dt = 'train'
                    elif counter < np.floor(np.sum(self.split)*total):
                        dt = 'valid'
                    else:
                        dt = 'test'

                    data['dt'] = dt
                counter += 1

                yield data

    def _report(self, dl):
        report = '''General information
                Number of utterances: %d
                Total size (in seconds) of utterances: %.f
                Number of speakers: %d''' % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])))

        return report

    def __str__(self):
        return self._name
