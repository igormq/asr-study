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
    def __init__(self, epsilon=1e-5, num_var=1, weights=None, gain_init='one',
                 bias_init='zero', **kwargs):
        self.epsilon = epsilon
        self.num_var = num_var
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

        input_shape = self.input_spec[0].shape
        output_dim =  input_shape[-1] // self.num_var

        if self.num_var == 1:
            input_list = [x]
            g_list = [self.g]
            b_list = [self.b]
        else:
            # input_list = [x[i*output_dim:(i+1)*output_dim] for i in xrange(self.num_var)]
            # TODO: Fix this issue
            import tensorflow as tf
            input_list = tf.split(1, self.num_var, x)

            g_list = [self.g[i*output_dim:(i+1)*output_dim] for i in xrange(self.num_var)]
            b_list = [self.b[i*output_dim:(i+1)*output_dim] for i in xrange(self.num_var)]

        outputs = []
        for n in xrange(self.num_var):
            x = input_list[n]
            g = g_list[n]
            b = b_list[n]

            m, std = self._moments(x)
            x_normed = (x - m) / (std + self.epsilon)

            output = g*x_normed + b

            outputs.append(output)


        if self.num_var == 1:
            return outputs[0]
        else:
            return K.concatenate(outputs)

    def _moments(self, x):
        '''
        # Arguments
            x: matrix [batch_size, output_dim]
        '''
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        return m, std

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
    def __init__(self, output_dim, nb_layers=1,
                 init='glorot_uniform', inner_init='orthogonal',
                 bias_init=highway_bias_initializer,
                 activation='tanh', inner_activation='hard_sigmoid',
                 coupling=True, layer_norm=False, W_regularizer=None, U_regularizer=None,
                 b_regularizer=None, dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.nb_layers = nb_layers
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.bias_init = initializations.get(bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.coupling = coupling
        self.has_layer_norm = layer_norm
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True

        self.layer_norm = None
        if self.has_layer_norm:
            # Should layer norm be applied to all the pre activations?
            # self.layer_norm = LayerNormalization(num_var=(2 + (not self.coupling)))
            self.layer_norm = LayerNormalization()

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
        self.U = self.inner_init((self.output_dim, (2 + (not self.coupling)) * self.output_dim),
                                 name='{}_U'.format(self.name))

        bias_init_value = K.get_value(self.bias_init((self.output_dim,)))
        b = [np.zeros(self.output_dim),
             np.copy(bias_init_value)]

        if not self.coupling:
            b.append(np.copy(bias_init_value))

        self.b = K.variable(np.hstack(b),
                            name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.U, self.b]

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
        B_U_W = states[1]

        for layer in xrange(self.nb_layers):
            B_U = B_U_W[layer][0]

            if layer == 0:
                B_W = B_U_W[layer][1]
                a = K.dot(x * B_W, self.W) + K.dot(s_tm1 * B_U, self.U) + self.b
            else:
                a = K.dot(s_tm1 * B_U, self.U) + self.b

            # LN should be applied to all activation or only to activation?
            # if self.has_layer_norm:
                # a = self.layer_norm(a)

            a0 = a[:, :self.output_dim]
            if self.has_layer_norm:
                a0 = self.layer_norm(a0)

            a1 = a[:, self.output_dim: 2 * self.output_dim]
            if not self.coupling:
             a2 = a[:, 2 * self.output_dim:]

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

        for layer in xrange(self.nb_layers):
            constant = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.output_dim))
                B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(2 + (not self.coupling))]
                constant.append(B_U)
            else:
                constant.append([K.cast_to_floatx(1.) for _ in range(2 + (not self.coupling))])

            if layer == 0:
                if 0 < self.dropout_W < 1:
                    input_shape = self.input_spec[0].shape
                    input_dim = input_shape[-1]
                    ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                    ones = K.tile(ones, (1, input_dim))
                    B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(2 + (not self.coupling))]
                    constant.append(B_W)
                else:
                    constant.append([K.cast_to_floatx(1.) for _ in range(2 + (not self.coupling))])

            constants.append(constant)

        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'nb_layers': self.nb_layers,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'bias_init': self.bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'coupling': self.coupling,
                  'layer_norm': self.layer_norm,
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
    model.add(RHN(10, input_dim=2, layer_norm=True))
    plot(model)
