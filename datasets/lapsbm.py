from datasets import DatasetParser

import os
import re
import librosa
import codecs


class LapsBM(DatasetParser):
    """ Laps benchmark version 1.4 dataset reader and parser

    More about this dataset: http://www.laps.ufpa.br/falabrasil/downloads.php
    """

    version = '1.4'

    # Random separation of LAPSBM1.4 dataset in validation and test if required
    # 5 women, 10 men
    _test_speaker_id = [3, 11, 13, 17, 12,
                        33, 5, 22, 16, 8,
                        4, 0, 20, 10, 9]

    # 5 women, 15 men
    _valid_speaker_id = [29, 32, 14, 31, 25,
                         23, 19, 26, 6, 2,
                         24, 15, 1, 21, 28,
                         30, 34, 27, 18, 7]

    def __init__(self, dataset_dir=None, name='lapsbm', split=False, **kwargs):

        dataset_dir = dataset_dir or 'data/lapsbm'

        self._split = split

        super(LapsBM, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):
        for speaker_path in os.listdir(self.dataset_dir):

            root_path = os.path.join(os.path.abspath(self.dataset_dir),
                                     speaker_path)

            if not os.path.isdir(os.path.join(root_path)):
                continue

            label_files = [f for f in os.listdir(root_path)
                           if '.txt' in f.lower()]

            for label_file in label_files:

                label = ' '.join(
                    codecs.open(
                        os.path.join(root_path, label_file), 'r',
                        encoding='utf8')
                    .read().strip().split(' ')).lower()

                audio_file = os.path.join(root_path,
                                          "%s.wav" % (label_file[:-4]))
                gender_speaker = speaker_path.split('-')[1]
                gender = gender_speaker[0].lower()
                speaker_id = gender_speaker[1:]

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    print('File %s not found' % audio_file)
                    continue

                dataset = 'valid'
                if int(speaker_id) in self._test_speaker_id:
                    dataset = 'test'

                data = {'duration': duration,
                        'input': audio_file,
                        'label': label,
                        'gender': gender,
                        'speaker': speaker_id}

                if self._split:
                    data['dataset'] = dataset

                yield data

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           Number of speakers: %d'
           %% of female speaker: %.2f%%''' \
           % (len(dl['audio']), sum(dl['duration']), len(set(dl['speaker'])),
              100 * (sum([1 for g in dl['gender'] if g == 'f']) /
                        (1.0 * len(dl['gender']))))

        return report
