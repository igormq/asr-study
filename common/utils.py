from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import sys
import os

import logging
import logging.config
import yaml

import numpy as np
from scipy import sparse

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf

import inspect
import yaml

import core
from core import layers_utils
from core import ctc_utils
from core import metrics


from .hparams import HParams


logger = logging.getLogger(__name__)


def safe_mkdirs(path):
    ''' Safe makedirs
    Directory is created with command `makedir -p`.
    Returns:
        `path` if the directory already exists or is created
    Exception:
        OSError if something is wrong
    '''
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != 17:  # 17 = file exists
            raise

    return path


def get_from_module(module, name, params=None):
    members = inspect_module(module)

    if name is None or name.lower() == 'none':
        return None

    members = {k.lower().strip(): v for k, v in members.items()}

    try:
        member = members[name.lower().strip()]

        # is a class and must be instantiate if params is not none
        if (member and params) and inspect.isclass(member):
            if type(params) is str:
                return member(
                    **HParams(from_str=params).values())

            if type(params) in (dict):
                return member(**HParams(**params).values())

            raise ValueError("params was not recognized.")

        return member
    except KeyError, e:
        raise KeyError("%s not found in %s.\n Valid values are: %s" %
                       (name, module, ', '.join(members.keys())))


def inspect_module(module, to_dict=True):
    members = inspect.getmembers(sys.modules[module], lambda member:
                                 hasattr(member, '__module__') and
                                 member.__module__ == module)
    if to_dict:
        return dict(members)

    return members


def get_custom_objects():
    """ Verify all custom object that may be used to load a keras model
    """
    all_custom_objects = []
    for module in ['core.layers', 'core.layers_utils',
                   'core.metrics', 'core.ctc_utils']:
        all_custom_objects.extend(inspect_module(module, to_dict=False))

    return dict(all_custom_objects)


def load_meta(model_fname):
    meta = {}

    try:
        with h5py.File(model_fname, 'r') as f:
            meta_group = f['meta']

            meta['training_args'] = yaml.load(
                meta_group.attrs['training_args'])
            for k in meta_group.keys():
                meta[k] = list(meta_group[k])
    except KeyError:
        # Tries to load the yaml file
        meta_fname = os.path.join(os.path.split(model_fname)[0], 'meta.yaml')
        if not os.path.isfile(meta_fname):
            raise Exception("Meta information was not found")

        with open(meta_fname, 'r') as f:
            meta = yaml.load(f)

    return meta


def load_model(model_fname, return_meta=False, mode='train'):
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
    dec = Lambda(ctc_utils.decode,
                 output_shape=ctc_utils.decode_output_shape,
                 arguments={'is_greedy': False,
                            'beam_width': 400}, name='beam_search')

    if mode == 'predict':
        y_pred = (model.get_layer('y_pred') or
                  model.get_layer('decoder').input[0])

        input_ = model.get_layer('input').input
        input_length = model.get_layer('input_length').input

        to_dense_layer = Lambda(
            layers_utils.to_dense,
            output_shape=layers_utils.to_dense_output_shape,
            name="to_dense")

        y_pred = dec([y_pred, input_length])

        y_pred = to_dense_layer(y_pred)

        model = Model(input=[input_, input_length],
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


def ld2dl(ld):
    '''Transform a list of dictionaries in a dictionaries with lists
    # Note
        All dictionaries have the same keys
    '''
    return dict(zip(ld[0], zip(*[d.values() for d in ld])))


def config_gpu(gpu, allow_growth=False, log_device_placement=False):
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


def parse_nondefault_args(args, default_args):
    # removing default arguments
    args_default = {k: v for k, v in vars(default_args).items()
                    if k not in [arg.split('-')[-1] for arg in sys.argv
                                 if arg.startswith('-')]}
    args_nondefault = {k: v for k, v in vars(args).items()
                       if k not in args_default or args_default[k] != v}

    args_nondefault = HParams(from_str=str(args_nondefault))

    return args_nondefault


def setup_logging(default_path='logging.yaml', default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def check_ext(fname, ext):
    # Adding dot
    ext = ext if ext[0] == '.' else '.' + ext
    fname, f_ext = os.path.splitext(fname)

    if f_ext == ext:
        return True

    return False
