from datasets import DatasetParser

import os
import re
import librosa
import codecs

class CSLUSpoltechPort(DatasetParser):

    def _iter(self):
        trans_directory = os.path.join(self.dt_dir, 'trans')

        for speaker_path in os.listdir(trans_directory):

            root_path = os.path.join(os.path.abspath(trans_directory), speaker_path)

            if not os.path.isdir(os.path.join(root_path)):
                continue

            labels_files = os.listdir(root_path)

            for labels_file in labels_files:

                label = codecs.open(os.path.join(root_path, labels_file), 'r', 'latin-1').read().strip().lower()
                audio_file = os.path.join(os.path.abspath(self.dt_dir), 'speech', speaker_path, labels_file[:-4])

                audio_file = audio_file + '.wav'
                speaker_id = speaker_path

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    print('File %s not found' % audio_file)
                    continue

                if not self._is_valid_label(label):
                    print(u'File %s has a forbidden label: "%s". Skipping' % (audio_file, label))
                    continue

                yield {'duration': duration,
                       'audio': audio_file,
                       'label': label,
                       'speaker': speaker_id}

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           Number of speakers: %d''' % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])))

        return report

    def __str__(self):
        return 'cslu'
