# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

import keras.backend as K
from keras.layers.recurrent import Recurrent
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec

import warnings

def highway_bias_initializer(shape, name=None):
    return -2*initializations.one(shape, name=name)


def layer_normalization(x, gain, bias, epsilon=1e-5):
    m = K.mean(x, axis=-1, keepdims=True)
    std = K.std(x, axis=-1, keepdims=True)
    x_normed = (x - m) / (std + epsilon) * gain + bias
    return x_normed

class LayerNormalization(Layer):
    '''Normalize from all of the summed inputs to the neurons in a layer on
    a single training case. Unlike batch normalization, layer normalization
    performs exactly the same computation at training and tests time.

    # Arguments
        epsilon: small float > 0. Fuzz parameter
        num_var: how many tensor are condensed in the input
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gain, bias]
        gain_init: name of initialization function for gain parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        bias_init: name of initialization function for bias parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    # Input shape

    # Output shape
        Same shape as input.

    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    '''
    def __init__(self, epsilon=1e-5, weights=None, gain_init='one',
                 bias_init='zero', **kwargs):
        self.epsilon = epsilon
        self.gain_init = initializations.get(gain_init)
        self.bias_init = initializations.get(bias_init)
        self.initial_weights = weights

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[-1],)

        self.g = self.gain_init(shape, name='{}_gain'.format(self.name))
        self.b = self.bias_init(shape, name='{}_bias'.format(self.name))

        self.trainable_weights = [self.g, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        return layer_normalization(x, self.g, self.b, epsilon=self.epsilon)

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  'num_var': self.num_var,
                  'gain_init': self.gain_init.__name__,
                  'bias_init': self.bias_init.__name__}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RHN(Recurrent):
    '''Recurrent Highway Network - Julian Georg Zilly, Rupesh Kumar Srivastava,
    Jan Koutník, Jürgen Schmidhuber - 2016.
    For a step-by-step description of the network, see
    [this paper](https://arxiv.org/abs/1607.03474).
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        depth: recurrency depth size.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        bias_init: initialization function of the bias.
            (see [this post](http://people.idsia.ch/~rupesh/very_deep_learning/)
            for more information)
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        coupling: if True, carry gate will be coupled to the transform gate,
            i.e., c = 1 - t
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    # References
        - [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474) (original paper)
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    # TODO: different dropout rates for each layer
    '''
    def __init__(self, output_dim, depth=1,
                 init='glorot_uniform', inner_init='orthogonal',
                 bias_init=highway_bias_initializer,
                 activation='tanh', inner_activation='hard_sigmoid',
                 coupling=True, layer_norm=False, ln_gain_init='one', ln_bias_init='zero',
                 W_regularizer=None, U_regularizer=None,
                 b_regularizer=None, dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.depth = depth
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.bias_init = initializations.get(bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.coupling = coupling
        self.has_layer_norm = layer_norm
        self.ln_gain_init = initializations.get(ln_gain_init)
        self.ln_bias_init = initializations.get(ln_bias_init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True

        super(RHN, self).__init__(**kwargs)

        if not self.consume_less == "gpu":
            warnings.warn("Ignoring consume_less=%s. Setting to 'gpu'." % self.consume_less)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None]


        self.W = self.init((self.input_dim, (2 + (not self.coupling)) * self.output_dim),
                           name='{}_W'.format(self.name))
        self.Us = [self.inner_init((self.output_dim, (2 + (not self.coupling)) * self.output_dim),
                                    name='%s_%d_U' %(self.name, i)) for i in xrange(self.depth)]

        bias_init_value = K.get_value(self.bias_init((self.output_dim,)))
        b = [np.zeros(self.output_dim),
             np.copy(bias_init_value)]

        if not self.coupling:
            b.append(np.copy(bias_init_value))

        self.bs = [K.variable(np.hstack(b), name='%s_%d_b' %(self.name, i)) for i in xrange(self.depth)]

        self.trainable_weights = [self.W] + self.Us +  self.bs
        if self.has_layer_norm:
            self.ln_weights = []
            ln_names = ['h', 't', 'c']
            for l in xrange(self.depth):
                # ln_gains = [self.ln_gain_init((self.output_dim,), name='%s_%d_ln_gain_%s' %(self.name, l, ln_names[i])) for i in xrange(2 + (not self.coupling))]
                ln_gains = [self.ln_gain_init((self.output_dim,), name='%s_%d_ln_gain_%s' %(self.name, l, ln_names[i])) for i in xrange(1)]
                # ln_biases = [self.ln_bias_init((self.output_dim,), name='%s_%d_ln_bias_%s' %(self.name, l, ln_names[i])) for i in xrange(2 + (not self.coupling))]
                ln_biases = [self.ln_bias_init((self.output_dim,), name='%s_%d_ln_bias_%s' %(self.name, l, ln_names[i])) for i in xrange(1)]
                self.ln_weights.append([ln_gains, ln_biases])
                self.trainable_weights += ln_gains + ln_biases

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        s_tm1 = states[0]

        for layer in xrange(self.depth):
            B_U = states[layer + 1][0]
            U, b = self.Us[layer], self.bs[layer]

            if layer == 0:
                B_W = states[layer + 1][1]
                a = K.dot(x * B_W, self.W) + K.dot(s_tm1 * B_U, U) + b
            else:
                a = K.dot(s_tm1 * B_U, U) + b

            # LN should be applied to all activation or only to activation?
            # if self.has_layer_norm:
                # a = self.layer_norm(a)

            a0 = a[:, :self.output_dim]
            a1 = a[:, self.output_dim: 2 * self.output_dim]
            if not self.coupling:
                a2 = a[:, 2 * self.output_dim:]

            if self.has_layer_norm:
                ln_gains, ln_biases = self.ln_weights[layer]
                a0 = layer_normalization(a0, ln_gains[0], ln_biases[0])
                # a1 = layer_normalization(a1, ln_gains[1], ln_biases[1])
                # if not self.coupling:
                #     a2 = layer_normalization(a2, ln_gains[2], ln_biases[2])

            # Equation 7
            h =  self.activation(a0)
            # Equation 8
            t = self.inner_activation(a1)
            #Equation 9
            if not self.coupling:
                c = self.inner_activation(a2)
            else:
                c = 1 - t # carry gate was coupled to the transform gate

            s  = h * t + s_tm1 * c
            s_tm1 = s


        return s, [s]

    def get_constants(self, x):
        constants = []

        for layer in xrange(self.depth):
            constant = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.output_dim))
                B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
                constant.append(B_U)
            else:
                constant.append(K.cast_to_floatx(1.))

            if layer == 0:
                if 0 < self.dropout_W < 1:
                    input_shape = self.input_spec[0].shape
                    input_dim = input_shape[-1]
                    ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                    ones = K.tile(ones, (1, input_dim))
                    B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
                    constant.append(B_W)
                else:
                    constant.append(K.cast_to_floatx(1.))

            constants.append(constant)

        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'depth': self.depth,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'bias_init': self.bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'coupling': self.coupling,
                  'layer_norm': self.has_layer_norm,
                  'ln_gain_init': self.ln_gain_init.__name__,
                  'ln_bias_init': self.ln_bias_init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(RHN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    from keras.models import Sequential
    from keras.utils.visualize_util import plot

    model = Sequential()
    model.add(RHN(10, input_dim=2, depth=2, layer_norm=True))
    # plot(model)
