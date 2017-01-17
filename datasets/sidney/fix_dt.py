from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import re
import wave
import codecs
import numpy as np
import fnmatch
from shutil import copyfile

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

        new_filepath = os.path.join(output_directory, root, filename.lower())
        copyfile(filepath, new_filepath)
