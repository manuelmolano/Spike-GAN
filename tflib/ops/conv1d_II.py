#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:01:32 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:56:13 2017

@author: manuel
"""

import tflib as lib
import numpy as np
import tensorflow as tf

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Conv1D(name, input_dim, output_dim, filter_size, inputs, he_init=True, stride=1, save_filter=False):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name):

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')


        
        if _weights_stdev is not None:
            filter_values = uniform(_weights_stdev,(filter_size, input_dim, output_dim))
        else:
            fan_in = input_dim * filter_size
            fan_out = output_dim * filter_size / stride

            if he_init:
                filters_stdev = np.sqrt(4./(fan_in+fan_out))
            else: # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2./(fan_in+fan_out))

            filter_values = uniform(filters_stdev,(filter_size, input_dim, output_dim))
        
     
        filters = lib.param(name+'.Filters', filter_values)
        result = tf.nn.conv1d(value=inputs, filters=filters, stride=stride, padding='SAME', data_format='NCHW')

        
        _biases = lib.param(name+'.Biases', np.zeros([output_dim], dtype='float32'))

       
        result = tf.expand_dims(result, 3)
        result = tf.nn.bias_add(result, _biases, data_format='NCHW')
        result = tf.squeeze(result)
        if save_filter:
            return result, filters
        else:
            return result
        

