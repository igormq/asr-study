from datasets import DatasetParser

import os
import re
import librosa
import codecs
import tempfile

import numpy as np

class Dummy(DatasetParser):

    def __init__(self, dt_dir, nb_speakers=10, nb_utterances_per_speaker=10, max_duration=10.0, min_duration=1.0, max_label_length=200, fs=16e3):
        super(Dummy, self).__init__(dt_dir)

        self.nb_speakers = nb_speakers
        self.nb_utterances_per_speaker = nb_utterances_per_speaker
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.fs = fs
        self.max_label_length = max_label_length

    def _iter(self):
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

                yield {'duration': duration,
                       'audio': audio_fname,
                       'label': label,
                       'speaker': 'speaker_%d' % speaker}

    def _report(self, dl):
        report = '''General information
                Number of utterances: %d
                Total size (in seconds) of utterances: %.f
                Number of speakers: %d''' % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])))

        return report

    def __str__(self):
        return 'dummy'
