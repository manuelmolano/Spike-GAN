# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:30:39 2017

@author: manuel
"""
import tensorflow as tf
from tflib.ops import linear
import time
import numpy as np
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLU(x):
    return tf.nn.relu(x)

def ReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)



if __name__ == '__main__':
    
    start = time.time()
    num_features = 64
    kernel_width = 5
    inputs = np.random.random(size=(64,1, num_features,16)).astype('float32')
    
    
    aux = ReLULayer('test', 512, 512, inputs)
    print(aux.get_shape())
    aux = tf.transpose(aux, [0, 2, 1, 3])
    print(aux.get_shape())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for ind in range(10000):
        test2 = sess.run(aux)
    print(time.time()-start)
    print(test2.shape)
    
    
    
    
    
    
    