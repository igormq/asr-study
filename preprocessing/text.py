# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import string
from unidecode import unidecode
import numpy as np

PUNCTUATIONS = "'""-,.!?:;"
ACCENTS = 'ãõçâêôáéíóúà'

class CharParser(object):
    ''' Class responsible to map any text in a certain vocabulary
    '''

    def __init__(self, mode='blank'):
        ''' Constructor of the class
        Arguments:
            mode: Which type of vacabulary will be generated. Modes can be concatenated by using pipeline '|'
                'blank' or 'b': accepts space character
                'accents' or 'a': accepts pt-br accents
                'punctuation' or 'p': accepts punctuation defined in string.punctuation
                'digits': accepts all digits
                'sensitive' or 's': characters will be case sensitive
                'all': shortcut that enables all modes
        '''
        self._permitted_modes = {'sensitive': 's', 'blank': 'b', 'accents': 'a', 'punctuation': 'p', 'digits': 'd'}

        if mode == 'all':
            self.mode = self._permitted_modes.values()
        else:
            self.mode = []
            for m in mode.split('|'):
                try:
                    self.mode.append(self._permitted_modes[m])
                except KeyError:
                    if m not in self._permitted_modes.values():
                        raise ValueError('Unknown mode %s' % m)

                    self.mode.append(m)

        self._vocab, self._inv_vocab = self._gen_vocab()

    def map(self, txt):
        label = np.array([self._vocab[c] for c in self._sanitize(txt)], dtype='int32')

        return label

    def imap(self, labels):
        txt = ''.join([self._inv_vocab[l] for l in labels])

        return txt

    def _sanitize(self, text):
        # removing duplicated spaces
        text = ' '.join(text.split())

        if not('d' in self.mode):
            text = ''.join([c for c in text if not c.isdigit()])

        if not('a' in self.mode):
            text = unidecode(text)

        if not('p' in self.mode):
            text = text.translate(string.maketrans("-'", '  ')).translate(None, string.punctuation)

        if not ('b' in self.mode):
            text = text.replace(' ', '')

        if not('s' in self.mode):
            text = text.lower()

        return text

    def _gen_vocab(self):

        vocab = {chr(value + ord('a')): (value) for value in xrange(ord('z') - ord('a') + 1)}

        if 's' in self.mode:
            for char in vocab.keys():
                vocab[char.upper()] = len(vocab)

        if 'b' in self.mode:
            # Inserts space label
            vocab[' '] = len(vocab)

        if 'p' in self.mode:
            for p in PUNCTUATIONS:
                vocab[p] = len(vocab)

        if 'd' in self.mode:
            for num in range(10):
                vocab[str(num)]  = len(vocab)

        inv_vocab = {v: k for (k, v) in vocab.iteritems()}

        # Add blank label
        inv_vocab[len(inv_vocab)] = '<b>'

        return vocab, inv_vocab

simple_char_parser = lambda txt: CharParser().map(txt)
complex_char_parser = lambda txt: CharParser(mode='b|p|a|d').map(txt)
