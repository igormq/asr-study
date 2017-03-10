from datasets import DatasetParser

import os
import re
import librosa
import codecs

import numpy as np

regex = r"Nome=(?P<name>.*)[\n]+Idade=(?P<age>.*)[\n]+.*[\n]+Sexo=(?P<gender>.*)[\n]+Escolaridade=(?P<education>.*)[\n]+"


class Sid(DatasetParser):
    """ Sid dataset reader and parser
    """

    def __init__(self, dataset_dir=None, name='sid', **kwargs):

        dataset_dir = dataset_dir or 'data/sid'

        super(Sid, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):
        for speaker_path in os.listdir(self.dataset_dir):

            root_path = os.path.join(os.path.abspath(self.dataset_dir),
                                     speaker_path)

            if not os.path.isdir(os.path.join(root_path)):
                continue

            labels_file = os.path.join(root_path, 'prompts.txt')

            speaker_info_file = os.path.join(root_path, 'speaker.txt')

            with open(speaker_info_file) as f:
                info_text = f.read()

            pattern = re.compile(regex, re.MULTILINE | re.UNICODE)

            info = list(re.finditer(pattern, info_text))[0].groupdict()

            gender = info['gender'][0].lower()
            speaker_id = speaker_path.lower()

            try:
                age = int(info['age'])
            except ValueError:
                self._logger.error('age %s could not be converted in int.',
                                  (info['age']))
                age = 0

            for line in codecs.open(labels_file, 'r', encoding='utf8'):

                split = line.strip().split('=')
                file_id = int(split[0])

                label = split[1].lower()

                audio_file = os.path.join(
                    root_path, "%s%03d" % (speaker_path, file_id)) + '.wav'

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'gender': gender,
                       'speaker': speaker_id,
                       'age': age}

    def _report(self, dl):
        args = len(dl['audio']), sum(dl['duration']),
        len(set(dl['speaker'])),
        100 * (sum([1 for g in dl['gender'] if g == 'f']) /
               (1.0 * len(dl['gender']))),
        min([a for a in dl['age'] if a is not 0]),
        max(dl['age']), np.mean([a for a in dl['age'] if a is not 0])

        report = '''General information
                Number of utterances: %d
                Total size (in seconds) of utterances: %.f
                Number of speakers: %d
                %% of female speaker: %.2f%%
                age range: from %d to %d. Mean: %.f''' % (args)

        return report


if __name__ == '__main__':
    """ Script to fix some errors in sid dataset about the name convention
    on folder and some errors in transcription
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_directory', type=str,
                        help='Path to data directory')
    args = parser.parse_args()

    data_directory = args.data_directory
    output_directory = args.output_directory

    # fix wav filenamess
    matches = []
    for root, dirnames, filenames in os.walk(data_directory):
        for filename in fnmatch.filter(filenames, '*.[Ww][Aa][Vv]'):
            filepath = os.path.join(root, filename)
            number = "%03d" % int(filename[-7:-4])
            prefix = filepath.split(os.path.sep)[-2]

            new_filename = "%s%s" % (prefix, number) + '.wav'
            new_filepath = os.path.join(output_directory, root, new_filename)

            if not os.path.exists(os.path.join(output_directory, root)):
                os.makedirs(os.path.join(output_directory, root))

            copyfile(filepath, new_filepath)

    for root, dirnames, filenames in os.walk(data_directory):
        for filename in fnmatch.filter(filenames, '*.[tT][xX][tT]'):
            filepath = os.path.join(root, filename)

            if filename.lower().startswith('texto'):
                filename = 'prompts.txt'

            new_filepath = os.path.join(output_directory,
                                        root, filename.lower())
            copyfile(filepath, new_filepath)
