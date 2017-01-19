from datasets import DatasetParser

import os
import re
import librosa
import codecs

class LapsBM(DatasetParser):

    version = '1.4'

    def _iter(self):
        for speaker_path in os.listdir(self.dt_dir):

            root_path = os.path.join(os.path.abspath(self.dt_dir), speaker_path)

            if not os.path.isdir(os.path.join(root_path)):
                continue

            label_files = [f for f in os.listdir(root_path) if '.txt' in f.lower()]

            for label_file in label_files:

                label = ' '.join(codecs.open(os.path.join(root_path, label_file), 'r', encoding='utf8').read().strip().split(' ')).lower()

                audio_file = os.path.join(root_path, "%s.wav" % (label_file[:-4]))
                gender_speaker = speaker_path.split('-')[1]
                gender = gender_speaker[0].lower()
                speaker_id = gender_speaker[1:]

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    print('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'audio': audio_file,
                       'label': label,
                       'gender': gender,
                       'speaker': speaker_id}

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           Number of speakers: %d'
           %% of female speaker: %.2f%%''' % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])), 100*(sum([1 for g in genders if g == 'f']) / (1.0*len(dl['gender']))))

        return report

    def __str__(self):
        return 'lapsbm'
