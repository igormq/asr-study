from datasets import DatasetParser

import os
import re
import librosa
import codecs
import tempfile

import numpy as np


class Dummy(DatasetParser):
    """ Fake dataset reader and parser to do some tests

    # Arguments
        num_speakers: number of speakers
        num_utterances_per_speaker: number of utterances that each speaker will
        have
        max_duration: max duration in seconds of each fake audio
        min_duration: min duration in seconds of each fake audio
        max_label_length: max size of each fake label
        fs: sampling frequency of each fake audio
        split: list with two values. It will divide this dataset in three sets
        (train, valid and test) given the proportions
    """

    def __init__(self, num_speakers=10, num_utterances_per_speaker=10,
                 max_duration=10.0, min_duration=1.0, max_label_length=200,
                 fs=16e3, split=None, **kwargs):
        '''
        Args:
            split: list or nparray of size 2 that splits the data between
            train, valid and test. example: split = [.8 .15] = 80% train, 15%
            valid and 5% test
        '''

        kwargs.setdefault('name', 'dummy')

        super(Dummy, self).__init__(**kwargs)

        self.num_speakers = num_speakers
        self.num_utterances_per_speaker = num_utterances_per_speaker
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.fs = fs
        self.max_label_length = max_label_length
        self.split = split

        if split is not None and (len(split) != 2 or np.sum(split) > 1.):
            raise ValueError('Split must have len = 2 and must sum <= 1')

    def _iter(self):

        counter = 0
        total = self.num_speakers * self.num_utterances_per_speaker

        for speaker in range(self.num_speakers):
            for utterance in range(self.num_utterances_per_speaker):

                duration = np.random.uniform(low=self.min_duration,
                                             high=self.max_duration)

                samples = np.floor(duration * self.fs)
                audio = np.random.randn(int(samples))

                audio_file = tempfile.NamedTemporaryFile(delete=False)
                audio_fname = audio_file.name
                audio_file.close()

                librosa.output.write_wav(audio_fname, audio, self.fs)

                label = np.random.randint(
                    low=ord('a'), high=ord('z'),
                    size=(np.random.randint(2, self.max_label_length),))

                label = ''.join([chr(l) for l in label])

                data = {'duration': duration,
                        'input': audio_fname,
                        'label': label,
                        'speaker': 'speaker_%d' % speaker}

                if self.split is not None:
                    if counter < np.floor(self.split[0] * total):
                        dataset = 'train'
                    elif counter < np.floor(np.sum(self.split) * total):
                        dataset = 'valid'
                    else:
                        dataset = 'test'

                    data['dataset'] = dataset
                counter += 1

                yield data

    def _report(self, dl):
        report = '''General information
                Number of utterances: %d
                Total size (in seconds) of utterances: %.f
                Number of speakers: %d''' % (len(dl['audio']),
                                             sum(dl['duration']),
                                             len(set(dl['speaker'])))

        return report
