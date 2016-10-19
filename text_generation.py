'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, SimpleRNN
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from layers import RHN
import uuid
import os
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser(description='Text generation toy example.')
parser.add_argument('--maxlen', type=int, default=40)
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--layer', type=str, choices=['lstm', 'rhn', 'rnn',
                                                  'gru'], default='lstm')
parser.add_argument('--nb_layers', type=int, default=1)
parser.add_argument('--layer_norm', action='store_true', default=False)
parser.add_argument('-o', '--output_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_epoch', type=int, default=10)
parser.add_argument('--nb_outer_it', type=int, default=6)

args = parser.parse_args()

if args.layer == 'lstm':
    RNN = LSTM
elif args.layer == 'gru':
    RNN = GRU
elif args.layer == 'rnn':
    RNN = SimpleRNN
elif args.layer == 'rhn':
    RNN = RHN

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = args.maxlen
step = args.step

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()

if args.layer == 'rhn':
    model.add(RHN(args.output_dim, input_shape=(maxlen, len(chars)),
                  layer_norm=args.layer_norm, nb_layers=args.nb_layers))
else:
    model.add(RNN(args.output_dim, input_shape=(maxlen, len(chars))))
    for l in xrange(args.nb_layers-1):
        model.add(RNN(args.output_dim))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
history = []
for _ in xrange(args.nb_outer_it):
    his = model.fit(X, y, batch_size=args.batch_size, nb_epoch=args.nb_epoch)
    his = his.history

    start_index = random.randint(0, len(text) - maxlen - 1)

    his['seed'] = text[start_index: start_index + maxlen]

    for diversity in [0.2, 0.5, 1.0, 1.2]:

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence


        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        his['diversity_%.1f' % diversity] = generated

    history.append(his)

meta = {'history': history, 'params': vars(args)}

if not os.path.isdir('results'):
    os.makedirs('results')

name = os.path.join('results', str(uuid.uuid1()))
print('Saving at ./%s' % name)
with open(name, 'wb') as f:
    pickle.dump(meta, f)

model.save('%s.h5' % name)
