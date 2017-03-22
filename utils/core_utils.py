from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import yaml

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf

import core
from core import layers_utils
from core import ctc_utils
from core import metrics

from utils.generic_utils import inspect_module


def setup_gpu(gpu, allow_growth=False, log_device_placement=False):
    # Choosing gpu
    if gpu == '-1':
        config = tf.ConfigProto(device_count={'GPU': 0},
                                log_device_placement=log_device_placement)
    else:
        if gpu == 'all':
            gpu = ''
        config = tf.ConfigProto(log_device_placement=log_device_placement)
        config.gpu_options.visible_device_list = gpu
    if allow_growth:  # dynamic gpu memory allocation
        config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


def get_custom_objects():
    """ Verify all custom object that may be used to load a keras model
    """
    all_custom_objects = []
    for module in ['core.layers', 'core.layers_utils',
                   'core.metrics', 'core.ctc_utils',
                   'core.initializers']:
        all_custom_objects.extend(inspect_module(module, to_dict=False))

    return dict(all_custom_objects)

def load_model(model_fname, return_meta=False, mode='train', **kwargs):
    """ Loading keras model with custom objects

    Args
        mode:
            if 'train', model will follow the definition in core.models
            if 'predict', beamsearch decoder will be used and the model return
            a np array with -1 filled in no data area
            if 'eval', greedy decoder will be replaced by beam search decoder
        of predictions
    """
    if mode not in ('train', 'predict', 'eval'):
        raise ValueError('mode must be one of (train, predict, eval)')

    model = keras.models.load_model(model_fname,
                                    custom_objects=get_custom_objects())

    # Define the new decoder and the to_dense layer
    if kwargs.get('decoder', True):
        dec = Lambda(ctc_utils.decode,
                     output_shape=ctc_utils.decode_output_shape,
                     arguments={'is_greedy': kwargs.get('is_greedy', False),
                                'beam_width': kwargs.get('beam_width', 400)},
                     name='beam_search')
    else:
        dec = Lambda(lambda x: x[0])

    if mode == 'predict':
        y_pred = (model.get_layer('y_pred') or
                  model.get_layer('decoder').input[0])

        input_ = model.get_layer('inputs').input
        inputs_length = model.get_layer('inputs_length').input

        to_dense_layer = Lambda(
            layers_utils.to_dense,
            output_shape=layers_utils.to_dense_output_shape,
            name="to_dense")

        y_pred = dec([y_pred, inputs_length])

        y_pred = to_dense_layer(y_pred)

        model = Model(input=[input_, inputs_length],
                      output=[y_pred])
    elif mode == 'eval':
        dec_layer = model.get_layer('decoder')

        y_pred_bs = dec(dec_layer.input)

        model = Model(input=model.inputs, output=[model.outputs[0], y_pred_bs])

        # Freezing layers
        for l in model.layers:
            l.trainable = False

        model.compile('sgd',
                      loss={'ctc': ctc_utils.ctc_dummy_loss,
                            'beam_search': ctc_utils.decoder_dummy_loss},
                      metrics={'beam_search': metrics.ler},
                      loss_weights=[1, 0])

    if return_meta:
        meta = load_meta(model_fname)
        return model, meta

    return model


def load_meta(model_fname):
    ''' Load meta configuration
    '''
    meta = {}

    with h5py.File(model_fname, 'r') as f:
        meta_group = f['meta']

        meta['training_args'] = yaml.load(
            meta_group.attrs['training_args'])
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])

    return meta
