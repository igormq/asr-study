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
    speakers = []

    trans_directory = os.path.join(data_directory, 'trans')

    for speaker_path in os.listdir(trans_directory):

        root_path = os.path.join(os.path.abspath(trans_directory), speaker_path)

        if not os.path.isdir(os.path.join(root_path)):
            continue

        labels_files = os.listdir(root_path)

        for labels_file in labels_files:

            label = codecs.open(os.path.join(root_path, labels_file), 'r', 'latin-1').read().strip().lower()
            audio_file = os.path.join(os.path.abspath(data_directory), 'speech',
                                      speaker_path, labels_file[:-4])


            audio_file = audio_file + '.wav'
            speaker = speaker_path

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
            speakers.append(speaker)

    with codecs.open(os.path.join(output_path, 'data.json'), 'w', encoding='utf-8') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i], 'speaker': speakers[i]}, ensure_ascii=False)
            out_file.write(line + '\n')

    with open(os.path.join(output_path, 'info.txt'), 'w') as f:
         s = '''General information:
            Number of utterances: %d
            Total size (in seconds) of utterances: %.f
            Number of speakers: %d''' % (len(keys), sum(durations), len(set(speakers)))

         print(s)
         f.write(s + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    args = parser.parse_args()

    main(args.data_directory, os.path.join(DT_ABSPATH, 'cslu_spoltech_port'))
