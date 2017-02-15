# NOTE: this example requires PyAudio because it uses the Microphone class
import sys
sys.path.insert(0, "/home/igormq/repos/asr-study/keras")

import os
import json
import argparse
import preprocessing
import inspect
import numpy as np

import speech_recognition as sr

import common.utils as utils
import common.apis as apis
from common.dataset_generator import DatasetIterator

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda

import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, nargs='+', default=['mic'])
    parser.add_argument('--language', default='pt-BR', type=str)

    # Custom asr
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')

    parser.add_argument('--apis', default=['google', 'ibm', 'microsoft'], nargs='+')

    args = parser.parse_args()

    r = sr.Recognizer()

    audios = []
    if len(args.source) == 1 and args.source[0] == 'mic':
        # obtain audio from the microphone
        with sr.Microphone() as source:
            print("Say something! (language %s)" % args.language)
            mic_audio = r.listen(source)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(mic_audio.get_wav_data())
        audios.append((f.name, 'microphone'))
    else:
        for audio_fname in args.source:
            with sr.AudioFile(audio_fname) as source:
                audios.append((r.record(source), audio_fname))
                # read the entire audio file

    if args.model is not None:
        utils.config_gpu(args.gpu, args.allow_growth)

        model, meta = utils.load_model(args.model,
                                       return_meta=True,
                                       mode='predict')
        training_args = meta['training_args']

        # Features extractor
        feats_extractor = utils.get_from_module('preprocessing.audio',
                                                training_args['feats'],
                                                training_args['feats_params'])

        # Recovering text parser
        text_parser = utils.get_from_module('preprocessing.text',
                                            training_args['text_parser'],
                                            training_args['text_parser_params']
                                            )

        data_it = DatasetIterator(np.array([f for a, f in audios]),
                                  feature_extractor=feats_extractor,
                                  text_parser=text_parser)

        model_predictions = model.predict_generator(
            data_it, val_samples=len(audios))

        model_predictions = [text_parser.imap(p[:(np.argmax(p == -1) or len(p))]) for p in model_predictions]

    for i, (audio, name) in enumerate(audios):

        print('Recognizing from: %s' % name)

        if 'google' in args.apis:
            rec = apis.recognize_google(audio, language=args.language)
            print("\tGoogle Cloud Speech:\n\t\t'%s'" % rec)

        if 'microsoft' in args.apis:
            rec = apis.recognize_bing(audio, language=args.language)
            print("\tMicrosoft Bing:\n\t\t'%s'" % rec)

        if 'ibm' in args.apis:
            rec = apis.recognize_ibm(audio, language=args.language)
            print("\tIBM Speech to Text:\n\t\t'%s'" % rec)

        if args.model is not None:
            print("\tTrained model:\n\t\t'%s'" % model_predictions[i])
