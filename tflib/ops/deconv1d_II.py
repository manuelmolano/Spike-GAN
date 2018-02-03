#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:30:11 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:12:26 2017

@author: manuel
"""

import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Deconv1D(name, input_dim, output_dim, filter_size, inputs, num_bins, he_init=True):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name):

      
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')
        stride = 2
        fan_in = input_dim * filter_size / stride
        fan_out = output_dim * filter_size


        if _weights_stdev is not None:
            filter_values = uniform(_weights_stdev, (filter_size, input_dim, output_dim))
        else:
            if he_init:
                filters_stdev = np.sqrt(4./(fan_in+fan_out))
            else: # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2./(fan_in+fan_out))
            filter_values = uniform(
                filters_stdev,
                (filter_size, input_dim, output_dim)
            )


        filters = lib.param(name+'.Filters', filter_values)        
       
        #input_shape = tf.shape(inputs)
        
        output_shape = tf.stack([1,  num_bins])#2*input_shape[2]])
#        print('--------------------------------')
#        print('inputs.get_shape()')
#        print(inputs.get_shape())
        aux_input = tf.expand_dims(inputs,axis=2)#to use the function resize_images we add an extra dimension to the input 
#        print('aux_input.get_shape()')
#        print(aux_input.get_shape())
        aux_input = tf.transpose(aux_input, [0,2,3,1], name='NCHW_to_NHWC')
        resized_image = tf.image.resize_images(images=aux_input, size=output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
#        print('resized_image.get_shape()')
#        print(resized_image.get_shape())
        resized_image = resized_image[:,0,:,:]#get rid of the extra dimension
        result = tf.nn.conv1d(value=resized_image, filters=filters, stride=1, padding='SAME')
#        print('result.get_shape()')
#        print(result.get_shape())
  
        _biases = lib.param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
        
        result = tf.expand_dims(result, 1)
#        print('after expansion')
#        print(result.get_shape())
        result = tf.nn.bias_add(result, _biases, data_format='NHWC')
#        print('after bias addition')
#        print(result.get_shape())
        result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')
#        print('after transpose')
#        print(result.get_shape())
        result = tf.squeeze(result)
#        print('after squeeze')
#        print(result.get_shape())
        
        return result
