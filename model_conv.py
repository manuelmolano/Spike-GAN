#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:17:31 2017

@author: manuel
"""

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from functools import wraps
import sys
sys.path.append(os.getcwd())
from tflib import plot, params_with_name, analysis
from tflib.ops import linear, act_funct, conv1d_II, deconv1d_II
from tensorflow.python.framework import ops as options
import matplotlib.pyplot as plt

#parameters used for figures
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

#not sure this is necessary
options.reset_default_graph()

def compatibility_decorator(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    name = kwds.pop('name', None)
    return f(targets=kwds['labels'], logits=kwds['logits'], name=name)
  return wrapper
   
# compatibility for TF v<1.0
if int(tf.__version__.split('.')[0]) < 1:
  tf.concat = tf.concat_v2
  tf.nn.sigmoid_cross_entropy_with_logits = compatibility_decorator(tf.nn.sigmoid_cross_entropy_with_logits)

class WGAN_conv(object):
  def __init__(self, sess, batch_size=64, lambd=10, stride=2, architecture = 'conv', num_units=512,
               num_neurons=4, z_dim=128, num_bins=32, num_layers=4, kernel_width=4, num_features=4,
               checkpoint_dir=None,
               sample_dir=None):  
    self.architecture = architecture #fully connected (fc) or convolutional (conv)
    self.num_units = num_units
    self.stride = stride
    self.sess = sess   
    self.batch_size = batch_size
    self.lambd = lambd #for the gradient penalization
    self.num_neurons = num_neurons
    self.num_bins = num_bins
    self.output_dim = self.num_neurons*self.num_bins #number of bins per samples
    self.z_dim = z_dim #latent space dimension
    self.num_layers = num_layers
    self.width_kernel = kernel_width # in the time dimension
    self.num_features = num_features #num features in the first layer of critic (this number will be duplicated in each succesive layer) [note: this is actually half of the number of features in the first generator layer]
    #folders
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    
    self.build_model()

  def build_model(self):
    #get the discriminator/generator corresponding to the selected architecture
    self.Discriminator, self.Generator, self.Discriminator_sampler = self.GeneratorAndDiscriminator()
    #real samples    
    self.inputs = tf.placeholder(tf.float32, name='real_data', shape=[self.batch_size, self.num_neurons*self.num_bins])
    #fake samples
    self.sample_inputs = self.Generator(self.batch_size,print_arch=True)
    self.ex_samples = self.get_samples()
    
    #discriminator output
    self.disc_real = self.Discriminator(self.inputs,print_arch=True)
    self.disc_fake = self.Discriminator(self.sample_inputs)

    #generator and discriminator cost
    self.gen_cost = -tf.reduce_mean(self.disc_fake)
    self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
    
    #penalize gradients
    alpha = tf.random_uniform(shape=[self.batch_size,1], minval=0., maxval=1.)
    differences = self.sample_inputs - self.inputs
    interpolates = self.inputs + (alpha*differences)
    aux = self.Discriminator(interpolates)
    gradients = tf.gradients(aux, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    self.disc_cost += self.lambd*gradient_penalty

    #this is to save the networks parameters
    self.saver = tf.train.Saver(max_to_keep=1000)

  def train(self, config):
    """Train DCGAN"""
    #define optimizer
    self.g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.gen_cost,
                                      var_list=params_with_name('Generator'), colocate_gradients_with_ops=True)
    self.d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.disc_cost,
                                       var_list=params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    tf.global_variables_initializer().run()
      
    #try to load trained parameters
    print('-------------')
    existing_gan, ckpt_name = self.load()
    
    #count number of variables
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('-------------')
    print('number of varaibles: ' + str(total_parameters))
    print('-------------')
    #start training
    counter_batch = 0
    epoch = 0
    #fitting errors
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for iteration in range(config.num_iter):
      start_time = time.time()
      # Train generator (only after the critic has been trained, at least once)
      if iteration+ckpt_name > 0:
         _ = self.sess.run(self.g_optim)
      
      # Train critic
      disc_iters = config.critic_iters
      for i in range(disc_iters):
        #get batch and update critic
        _data = self.training_samples[:,counter_batch*config.batch_size:(counter_batch+1)*config.batch_size].T
        _disc_cost, _ = self.sess.run([self.disc_cost, self.d_optim], feed_dict={self.inputs: _data})
        #if we have reached the end of the real samples set, we start over and increment the number of epochs
        if counter_batch==int(self.training_samples.shape[1]/self.batch_size)-1:
            counter_batch = 0
            epoch += 1
        else:
            counter_batch += 1
      aux = time.time() - start_time
      #plot the  critics loss and the iteration time
      plot.plot(self.sample_dir,'train disc cost', -_disc_cost)
      plot.plot(self.sample_dir,'time', aux)
    
      if (iteration+ckpt_name == 500) or iteration % 20000 == 19999 or (iteration+ckpt_name >= config.num_iter-10):
        print('epoch ' + str(epoch))
        if config.dataset=='uniform' or config.dataset=='packets':
            #this is to evaluate whether the discriminator has overfit 
            dev_disc_costs = []
            for ind_dev in range(int(self.dev_samples.shape[1]/self.batch_size)):
              images = self.dev_samples[:,ind_dev*config.batch_size:(ind_dev+1)*config.batch_size].T
              _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.inputs: images}) 
              dev_disc_costs.append(_dev_disc_cost)
            #plot the dev loss  
            plot.plot(self.sample_dir,'dev disc cost', -np.mean(dev_disc_costs))
        
        #save the network parameters
        self.save(iteration+ckpt_name)
        
        #get simulated samples, calculate their statistics and compare them with the original ones
        fake_samples = self.sess.run([self.ex_samples])[0]        
        acf_error, mean_error, corr_error, time_course_error,_ = analysis.get_stats(X=fake_samples.T, num_neurons=config.num_neurons,\
            num_bins=config.num_bins, folder=config.sample_dir, name='fake'+str(iteration+ckpt_name), critic_cost=-_disc_cost,instance=config.data_instance) 
        #plot the fitting errors
        sbplt[0][0].plot(iteration+ckpt_name,mean_error,'+b')
        sbplt[0][0].set_title('spk-count mean error')
        sbplt[0][0].set_xlabel('iterations')
        sbplt[0][0].set_ylabel('L1 error')
        sbplt[0][0].set_xlim([0-config.num_iter/4, config.num_iter+config.num_iter/4])
        sbplt[0][1].plot(iteration+ckpt_name,time_course_error,'+b')
        sbplt[0][1].set_title('time course error')
        sbplt[0][1].set_xlabel('iterations')
        sbplt[0][1].set_ylabel('L1 error')
        sbplt[0][1].set_xlim([0-config.num_iter/4, config.num_iter+config.num_iter/4])
        sbplt[1][0].plot(iteration+ckpt_name,acf_error,'+b')
        sbplt[1][0].set_title('AC error')
        sbplt[1][0].set_xlabel('iterations')
        sbplt[1][0].set_ylabel('L1 error')
        sbplt[1][0].set_xlim([0-config.num_iter/4, config.num_iter+config.num_iter/4])
        sbplt[1][1].plot(iteration+ckpt_name,corr_error,'+b')
        sbplt[1][1].set_title('corr error')
        sbplt[1][1].set_xlabel('iterations')
        sbplt[1][1].set_ylabel('L1 error')
        sbplt[1][1].set_xlim([0-config.num_iter/4, config.num_iter+config.num_iter/4])
        f.savefig(self.sample_dir+'fitting_errors.svg',dpi=600, bbox_inches='tight')
        plt.close(f)
        plot.flush(self.sample_dir)
    
      plot.tick()        
        
      
  def GeneratorAndDiscriminator(self):
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """
    if self.architecture=='conv':
        print('using convolutional achitecture')
        return self.DCGANDiscriminator, self.DCGANGenerator, self.DCGANDiscriminator_sampler     
    elif self.architecture=='fc':
        print('using fully connected achitecture')
        return self.FCDiscriminator, self.FCGenerator, self.FCDiscriminator_sampler     
  
  #####################convolutional GAN
  # Discriminator
  def DCGANDiscriminator(self, inputs, print_arch=False):
    kernel_width = self.width_kernel # in the time dimension
    num_features = self.num_features
    #neurons are treated as different channels
    output = tf.reshape(inputs, [-1, self.num_neurons, self.num_bins])
    conv1d_II.set_weights_stdev(0.02)
    deconv1d_II.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)
    if print_arch:
        print('DISCRIMINATOR. -------------------------------')
        print(str(output.get_shape())+' input')
    for ind_l in range(self.num_layers):
        if ind_l==0:
            output = conv1d_II.Conv1D('Discriminator.'+str(ind_l+1), self.num_neurons, num_features*2**(ind_l+1),kernel_width, output, stride=self.stride)
        else:
            output = conv1d_II.Conv1D('Discriminator.'+str(ind_l+1), num_features*2**(ind_l), num_features*2**(ind_l+1), kernel_width, output, stride=self.stride)
        output = act_funct.LeakyReLU(output)
        if print_arch:
            print(str(output.get_shape()) + ' layer '+ str(ind_l+1))
        
        
    output = tf.reshape(output, [-1, int(num_features*self.num_bins)])
    if print_arch:
        print(str(output.get_shape()) + ' fully connected layer')
    output = linear.Linear('Discriminator.Output', int(num_features*self.num_bins), 1, output)
    if print_arch:
        print(str(output.get_shape()) + ' output')
    conv1d_II.unset_weights_stdev()
    deconv1d_II.unset_weights_stdev()
    linear.unset_weights_stdev()

    return tf.reshape(output, [-1])
  
  
  
  def DCGANDiscriminator_sampler(self, inputs):
      kernel_width = self.width_kernel # in the time dimension
      num_features = self.num_features
      #neurons are treated as different channels
      output = tf.reshape(inputs, [-1, self.num_neurons, self.num_bins])
      #initialize weights
      conv1d_II.set_weights_stdev(0.02)
      deconv1d_II.set_weights_stdev(0.02)
      linear.set_weights_stdev(0.02)
      out_puts_mat = []
      filters_mat = []

      for ind_l in range(self.num_layers):
          if ind_l==0:
              output, filters = conv1d_II.Conv1D('Discriminator.'+str(ind_l+1), self.num_neurons, num_features*2**(ind_l+1),kernel_width, output, stride=self.stride, save_filter=True)
          else:
              output, filters = conv1d_II.Conv1D('Discriminator.'+str(ind_l+1), num_features*2**(ind_l), num_features*2**(ind_l+1), kernel_width, output, stride=self.stride, save_filter=True)
          output = act_funct.LeakyReLU(output)
          out_puts_mat.append(output)
          filters_mat.append(filters)
        
      output = tf.reshape(output, [-1, int(num_features*self.num_bins)])
      output = linear.Linear('Discriminator.Output', int(num_features*self.num_bins), 1, output)
      #unset weights
      conv1d_II.unset_weights_stdev()
      deconv1d_II.unset_weights_stdev()
      linear.unset_weights_stdev()


      return tf.reshape(output, [-1]), filters_mat, out_puts_mat
  
  #Generator
  def DCGANGenerator(self, n_samples, noise=None, print_arch=False):
      kernel_width = self.width_kernel # in the time dimension
      num_features = self.num_features
      conv1d_II.set_weights_stdev(0.02)
      deconv1d_II.set_weights_stdev(0.02)
      linear.set_weights_stdev(0.02)
      
      if noise is None:
          noise = tf.random_normal([n_samples, 128])
      if print_arch:
          print('GENERATOR. -------------------------------')
          print(str(noise.get_shape()) + ' latent variable')
      output = linear.Linear('Generator.Input', 128,int(num_features*self.num_bins), noise)
      if print_arch:
          print(str(output.get_shape()) + ' linear projection')
      output = tf.reshape(output, [-1, num_features*2**self.num_layers, int(self.num_bins/2**self.num_layers)])
      output = act_funct.LeakyReLU(output)
      if print_arch:
          print(str(output.get_shape()) + ' layer 1')
      for ind_l in range(self.num_layers,0,-1):
          if ind_l==1:
              output = deconv1d_II.Deconv1D('Generator.'+str(self.num_layers-ind_l+1), num_features*2**ind_l, self.num_neurons,\
                                            kernel_width, output, num_bins=int(2**(self.num_layers-ind_l+1)*self.num_bins/2**self.num_layers))
          else:
              output = deconv1d_II.Deconv1D('Generator.'+str(self.num_layers-ind_l+1), num_features*2**ind_l, num_features*2**(ind_l-1),\
                                            kernel_width, output, num_bins=int(2**(self.num_layers-ind_l+1)*self.num_bins/2**self.num_layers))
          output = act_funct.LeakyReLU(output)
          if print_arch:
              print(str(output.get_shape()) + ' layer ' + str(self.num_layers-ind_l+2))
      
   
      output = tf.sigmoid(output)

      conv1d_II.unset_weights_stdev()
      deconv1d_II.unset_weights_stdev()
      linear.unset_weights_stdev()
      output = tf.reshape(output, [-1, self.output_dim])
      
      return output
 
  #################fully connected GAN  
    
  # Discriminator
  def FCDiscriminator(self,inputs, n_layers=3):
    output = act_funct.LeakyReLULayer('Discriminator.Input', self.output_dim, self.num_units, inputs)
    for i in range(n_layers):
        output = act_funct.LeakyReLULayer('Discriminator.{}'.format(i), self.num_units, self.num_units, output)
    output = linear.Linear('Discriminator.Out', self.num_units, 1, output)

    return tf.reshape(output, [-1])
    
  # Discriminator
  def FCDiscriminator_sampler(self,inputs, n_layers=3):
      output = act_funct.LeakyReLULayer('Discriminator.Input', self.output_dim, self.num_units, inputs)
      outputs_mat = [output]
      for i in range(n_layers):
          output = act_funct.LeakyReLULayer('Discriminator.{}'.format(i), self.num_units, self.num_units, output)
          outputs_mat.append(output)
      output = linear.Linear('Discriminator.Out', self.num_units, 1, output)
      filters = []
      return tf.reshape(output, [-1]), filters, outputs_mat
  
  # Generator
  def FCGenerator(self, n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = act_funct.ReLULayer('Generator.1', 128, self.num_units, noise)
    output = act_funct.ReLULayer('Generator.2', self.num_units, self.num_units, output)
    output = act_funct.ReLULayer('Generator.3', self.num_units, self.num_units, output)
    output = act_funct.ReLULayer('Generator.4', self.num_units, self.num_units, output)
    output = linear.Linear('Generator.Out', self.num_units, self.output_dim, output)
    
    output = tf.nn.sigmoid(output)
    
    return output

  ######################auxiliary functions

  def binarize(self, samples, threshold=None):
    '''
    Returns binarized samples by thresholding with `threshold`. If `threshold` is `None` then the
    elements of `samples` are used as probabilities for drawing Bernoulli variates.
    '''
    if threshold is not None:
      binarized_samples = samples > threshold
    else:
      #use samples as probabilities for drawing Bernoulli random variates
      binarized_samples = samples > np.random.random(samples.shape)
    return binarized_samples.astype(float)  
  
  #draw samples from the generator
  def get_samples(self, num_samples=2**13): 
    #noise = tf.constant(np.random.normal(size=(num_samples, 128)).astype('float32'))
    fake_samples = self.Generator(num_samples)#, noise=noise)
    return fake_samples  

  #get filters from the discriminator
  def get_filters(self, num_samples=64):
      noise = tf.constant(np.random.normal(size=(num_samples, self.output_dim)).astype('float32'))
      _,filters,_ = self.Discriminator_sampler(noise)
  
      return filters
  
  #get units STA from the discriminator
  def get_units(self, num_samples):
      #aux = np.load(self.sample_dir+ '/stats_real.npz')
      noise = tf.constant(((np.zeros((num_samples, self.output_dim)) + 0.5) > np.random.random((num_samples, self.output_dim))).astype('float32'))
      output,_,units = self.Discriminator_sampler(noise)
      return output, units, noise  
  
  #get units STA from the discriminator
  def get_critics_output(self, samples):
      #aux = np.load(self.sample_dir+ '/stats_real.npz')
      output,_,_ = self.Discriminator_sampler(samples)
      return output
   
  #this is to save the network parameters  
  def save(self, step=0):
    model_name = "WGAN.model"
    self.saver.save(self.sess,os.path.join(self.checkpoint_dir, model_name),global_step=step)
    
  #this is to load an existing model
  def load(self, training_stage=''):
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, FOLDER)
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if training_stage=='':
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      else:
          #here we select a particular checkpoint by using ckpt.all_model_checkpoint_paths
          index = ckpt.all_model_checkpoint_paths[0].find('WGAN.model')
          index = ckpt.all_model_checkpoint_paths[0].find('-',index)
          for ind_ckpt in range(len(ckpt.all_model_checkpoint_paths)):
              counter = ckpt.all_model_checkpoint_paths[ind_ckpt][index+1:]
              if counter==training_stage:
                  ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[ind_ckpt])
                  break
    
      self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      index = ckpt_name.find('-')
      print(ckpt_name)
      print(int(ckpt_name[index+1:]))
      return True, int(ckpt_name[index+1:])
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0          

 
    
 
    
 