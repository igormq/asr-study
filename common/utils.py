from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import sys
import os

import numpy as np
from scipy import sparse
import keras

import inspect

import core

def get_from_h5(h5_file, dataset, label_type='char'):
    X = np.array(h5_file['%s/inputs/data' %dataset])
    seq_len = np.array(h5_file['%s/inputs/seq_len' %dataset])

    values = np.array(h5_file['%s/%s/values' %(dataset, label_type)])
    indices = np.array(h5_file['%s/%s/indices' %(dataset, label_type)])
    indices = (indices[:, 0], indices[:, 1])
    shape = np.array(h5_file['%s/%s/shape' %(dataset, label_type)])

    y = sparse.coo_matrix((values, indices), shape=shape).tolil()
    return X, seq_len, y

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
        if e.errno != 17: # 17 = file exists
            raise

    return path

def get_functions_from_module(module):
    return dict(inspect.getmembers(sys.modules[module], lambda member: inspect.isfunction(member) and member.__module__ == module))

def get_custom_objects():
    all_custom_objects = []

    for module in ['core.layers', 'core.metrics', 'core.ctc_utils']:
        all_custom_objects.extend(inspect.getmembers(sys.modules[module], lambda member: (inspect.isclass(member) or inspect.isfunction(member)) and member.__module__ == module))

    return dict(all_custom_objects)

def model_loader(treta_path):
    modelin = keras.models.load_model(treta_path, custom_objects=get_custom_objects())
    return modelin

def ld2dl(ld):
    '''Transform a list of dictionaries in a dictionaries with lists
    NOTE: All dictionaries have the same keys
    '''
    return dict(zip(ld[0],zip(*[d.values() for d in ld])))
