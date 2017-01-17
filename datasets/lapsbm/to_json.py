"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]

Where speaker.trans.txt has in each line, file_id transcription
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import re
import wave
import codecs
import numpy as np

from datasets import DT_ABSPATH
from datasets.common import utils


def main(data_directory, output_path):

    output_path = utils.mkdirs(output_path)

    labels = []
    durations = []
    keys = []
    genders = []
    speakers = []

    for speaker_path in os.listdir(data_directory):

        root_path = os.path.join(os.path.abspath(data_directory), speaker_path)

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
                audio = wave.open(audio_file)
            except IOError:
                print('File %s not found' % audio_file)
                continue

            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            keys.append(audio_file)
            durations.append(duration)
            labels.append(label)
            genders.append(gender)
            speakers.append(speaker_id)

    with codecs.open(os.path.join(output_path, 'data.json'), 'w', encoding='utf8') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i], 'gender': genders[i], 'speaker': speakers[i]}, ensure_ascii=False)
            out_file.write(line + '\n')

    with open(os.path.join(output_path, 'info.txt'), 'w') as f:
         s = '''General information:
            Number of utterances: %d
            Total size (in seconds) of utterances: %.f
            Number of speakers: %d'
            %% of female speaker: %.2f%%''' % (len(keys), sum(durations), len(set(speakers)), 100*(sum([1 for g in genders if g == 'f']) / (1.0*len(genders))))

         print(s)
         f.write(s + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')

    args = parser.parse_args()
    main(args.data_directory, os.path.join(DT_ABSPATH, 'lapsbm'))
