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

from datasets import DT_ABSPATH
from datasets.common import utils

ignored_list = ['Marcelo-20131106-iqc', 'anonymous-20140619-wcy', 'ThiagoCastro-20131129-qpn', 'anonymous-20131016-uzv']


def main(data_directory, output_path):

    output_path = utils.mkdirs(output_path)

    labels = []
    durations = []
    keys = []
    genders = []
    speakers = []
    for speaker_path in os.listdir(data_directory):

        if speaker_path in ignored_list:
            continue

        root_path = os.path.join(os.path.abspath(data_directory), speaker_path)


        if not os.path.isdir(os.path.join(root_path)):
            continue

        labels_file = os.path.join(root_path, 'etc', 'PROMPTS')

        if not os.path.exists(labels_file):
            labels_file = os.path.join(root_path, 'PROMPTS')

        speaker_info_file = os.path.join(root_path, 'etc', 'README')

        if not os.path.exists(speaker_info_file):
            speaker_info_file = os.path.join(root_path, 'README')


        with open(speaker_info_file) as f:
            info_text = f.read()

        pattern = re.compile(r"User\s+Name\:[\s]*(?P<speaker>.*)[\n]+.*[\n]+Gender\:[\s]*(?P<gender>[a-zA-Z]+)[\w\r\s\n:\/]+Pronunciation dialect\:\s+(?P<dialect>.*)", re.MULTILINE | re.UNICODE)

        info = list(re.finditer(pattern, info_text))[0].groupdict()

        gender = info['gender'][0].lower()
        speaker_id = info['speaker']

        for line in codecs.open(labels_file, 'r', encoding='utf-8'):
            split = line.strip().split()
            file_id = split[0].split('/')[-1]

            label = ' '.join(split[1:]).lower()

            audio_file = os.path.join(root_path, 'wav', file_id) + '.wav'

            if not os.path.exists(audio_file):
                audio_file = os.path.join(root_path, file_id) + '.wav'

            try:
                audio = wave.open(audio_file)
            except IOError:
                print('File %s not found' % audio)
                continue

            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            keys.append(audio_file)
            durations.append(duration)
            labels.append(label)
            genders.append(gender)
            speakers.append(speaker_id)

    with codecs.open(os.path.join(output_path, 'data.json'), 'w', encoding='utf-8') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i], 'gender': genders[i], 'speaker': speakers[i]}, ensure_ascii=False)
            out_file.write(line + '\n')

    with open(os.path.join(output_path, 'info.txt'), 'w') as f:
        s = '''General information
            Number of utterances: %d
            Total size (in seconds) of utterances: %.f
            Number of speakers: %d
            %% of female speaker: %.2f%%
            Anonymous speaker: %.2f%%''' % (len(keys), sum(durations), len(set(speakers)), 100*(sum([1 for g in genders if g == 'f']) / (1.0*len(genders))), 100*(sum([1 for s in speakers if s == 'anonymous']) / (1.0*len(speakers))))

        print(s)
        f.write(s + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    args = parser.parse_args()

    main(args.data_directory, os.path.join(DT_ABSPATH, 'voxforge'))
