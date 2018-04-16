#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:35:51 2017

@author: manuel
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:30:47 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:27:20 2017

@author: manuel
"""

import os
import pprint
from model_conv import WGAN_conv
from tflib import analysis, retinal_data, visualize_filters_and_units, data_provider, sim_pop_activity
import numpy as np
#from utils import pp, get_samples_autocorrelogram, get_samples


import tensorflow as tf
#parameters used for (some) figures

flags = tf.app.flags
flags.DEFINE_string("architecture", "conv", "semi-conv (conv) or fully connected (fc)")
flags.DEFINE_integer("num_iter", 300000, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam [1e-4]")
flags.DEFINE_float("beta1", 0., "Momentum term of adam [0.]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("recovery_dir", "", "in case the real samples are already stored in another folder")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("training_step", 200, "number of batches between weigths and performance saving")
flags.DEFINE_string("training_stage", '', "stage of the training used for the GAN")
flags.DEFINE_integer("num_layers", 2, "number of convolutional layers [4]")
flags.DEFINE_integer("num_features", 4, "features in first layers [4]")
flags.DEFINE_integer("kernel_width", 5, "width of kernel [4]")
flags.DEFINE_integer("num_units", 512, "num units per layer in the fc GAN")
flags.DEFINE_integer("critic_iters", 5, "number of times the discriminator will be updated")
flags.DEFINE_float("lambd", 10, "parameter gradient penalization")

#parameter set specifiying data
flags.DEFINE_string("sample_dir", "", "where the samples will be saved. This is automatically defined below, here I just initialize the field.")
flags.DEFINE_string("dataset", "uniform", "type of neural activity. It can be simulated  or retina")
flags.DEFINE_string("data_instance", "1", "if data==retina, this allows chosing the data instance")
flags.DEFINE_integer("num_samples", 2**13, "number of samples")
flags.DEFINE_integer("num_neurons", 4, "number of neurons in the population")
flags.DEFINE_float("packet_prob", 0.05, "probability of packets")
flags.DEFINE_integer("num_bins", 32, "number of bins (ms) for each sample")
flags.DEFINE_string("iteration", "0", "in case several instances are run with the same parameters")
flags.DEFINE_integer("ref_period", 2, "minimum number of ms between spikes (if < 0, no refractory period is imposed)")
flags.DEFINE_float("firing_rate", 0.25, "maximum firing rate of the simulated responses")
flags.DEFINE_float("correlation", 0.9, "correlation between neurons")
flags.DEFINE_integer("group_size", 4, "size of the correlated groups (e.g. 2 means pairwise correlations)")
flags.DEFINE_float("noise_in_packet", 0, "std of gaussian noise added to packets")
flags.DEFINE_integer("number_of_modes", 1, "[1,2] Number of different responses in the packet simulation. If =2 each type \
                     of packet will happen only once in the sample and one of two possible set of neurons will be chosen with equal prob for each packet.")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()
def main(_):
  #print parameters
  pp.pprint(tf.app.flags.FLAGS.flag_values_dict())
  #folders
  if FLAGS.dataset=='uniform':
      if FLAGS.architecture=='fc':
          FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
          + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
          '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
           '_num_units_' + str(FLAGS.num_units) +\
          '_iteration_' + FLAGS.iteration + '/'
      elif FLAGS.architecture=='conv':
          FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
          + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
          '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
          '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
          '_iteration_' + FLAGS.iteration + '/'
  elif FLAGS.dataset=='packets' and FLAGS.number_of_modes==1:
      if FLAGS.architecture=='fc':
          FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
          + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size) + '_critic_iters_' +\
          str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) + '_num_units_' + str(FLAGS.num_units) +\
          '_iteration_' + FLAGS.iteration + '/'
      elif FLAGS.architecture=='conv':
          FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
          + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size) + '_critic_iters_' +\
          str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
          '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
          '_iteration_' + FLAGS.iteration + '/'
  elif FLAGS.dataset=='packets' and FLAGS.number_of_modes==2:
      if FLAGS.architecture=='fc':
          FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
          + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size) + '_noise_in_packet_' + str(FLAGS.noise_in_packet) + '_number_of_modes_' + str(FLAGS.number_of_modes)  + '_critic_iters_' +\
          str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) + '_num_units_' + str(FLAGS.num_units) +\
          '_iteration_' + FLAGS.iteration + '/'
      elif FLAGS.architecture=='conv':
          FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
          + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size) + '_noise_in_packet_' + str(FLAGS.noise_in_packet) + '_number_of_modes_' + str(FLAGS.number_of_modes)  + '_critic_iters_' +\
          str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
          '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
          '_iteration_' + FLAGS.iteration + '/'
  elif FLAGS.dataset=='retina':
     if FLAGS.architecture=='fc':
          FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset  +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
          + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
          '_num_units_' + str(FLAGS.num_units) +\
          '_iteration_' + FLAGS.iteration + '/'
     elif FLAGS.architecture=='conv':
          FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
          '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
          + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
          '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
          '_iteration_' + FLAGS.iteration + '/'
      
  FLAGS.checkpoint_dir = FLAGS.sample_dir + 'checkpoint/'
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
  if FLAGS.recovery_dir=="" and os.path.exists(FLAGS.sample_dir+'/stats_real.npz'):
      FLAGS.recovery_dir = FLAGS.sample_dir
      
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  
  with tf.Session(config=run_config) as sess:
    wgan = WGAN_conv(sess, architecture=FLAGS.architecture,
        num_neurons=FLAGS.num_neurons,
        num_bins=FLAGS.num_bins,
        num_layers=FLAGS.num_layers, num_units=FLAGS.num_units,
        num_features=FLAGS.num_features,
        kernel_width=FLAGS.kernel_width,
        lambd=FLAGS.lambd,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)
    
    
    if FLAGS.is_train:  
        training_samples, dev_samples = data_provider.generate_spike_trains(FLAGS, FLAGS.recovery_dir)
        wgan.training_samples = training_samples
        wgan.dev_samples = dev_samples
        print('data loaded')
        wgan.train(FLAGS)
    else:
        if not wgan.load(FLAGS.training_stage):
            raise Exception("[!] Train a model first, then run test mode")      

    #LOAD TRAINING DATASET (and its statistics)
    original_dataset = np.load(FLAGS.sample_dir+ '/stats_real.npz')
    
    #PLOT FILTERS
    if FLAGS.dataset=='retina':
        index = np.arange(FLAGS.num_neurons)
    else:
        index = np.argsort(original_dataset['shuffled_index'])
        
    if FLAGS.architecture=='conv':
        print('get filters -----------------------------------')
        filters = wgan.get_filters(num_samples=64)
        visualize_filters_and_units.plot_filters(filters, sess, FLAGS, index)
    
    
    #GET GENERATED SAMPLES AND COMPUTE THEIR STATISTICS
    print('compute stats -----------------------------------')
    if 'samples' not in original_dataset:
        real_samples = retinal_data.get_samples(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance, folder=os.getcwd()+'/data/retinal data/')
    else:
        real_samples = original_dataset['samples']
    sim_pop_activity.plot_samples(real_samples, FLAGS.num_neurons, FLAGS.sample_dir, 'real')
    fake_samples = wgan.get_samples(num_samples=FLAGS.num_samples)
    fake_samples = fake_samples.eval(session=sess)
    sim_pop_activity.plot_samples(fake_samples.T, FLAGS.num_neurons, FLAGS.sample_dir, 'fake')
    _,_,_,_,_ = analysis.get_stats(X=fake_samples.T, num_neurons=FLAGS.num_neurons, num_bins= FLAGS.num_bins, folder=FLAGS.sample_dir, name='fake', instance=FLAGS.data_instance)


    #EVALUATE HIGH ORDER FEATURES (only when dimensionality is low)
    if FLAGS.dataset=='uniform' and FLAGS.num_bins*FLAGS.num_neurons<40:
        print('compute high order statistics-----------------------------------')
        num_trials = int(2**8)
        num_samples_per_trial = 2**13
        fake_samples_mat = np.zeros((num_trials*num_samples_per_trial,FLAGS.num_neurons*FLAGS.num_bins))
        for ind_tr in range(num_trials):
            fake_samples = wgan.sess.run([wgan.ex_samples])[0]   
            fake_samples_mat[ind_tr*num_samples_per_trial:(ind_tr+1)*num_samples_per_trial,:] = fake_samples
        
        analysis.evaluate_approx_distribution(X=fake_samples_mat.T, folder=FLAGS.sample_dir, num_samples_theoretical_distr=2**21,num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons,\
                          group_size=FLAGS.group_size,refr_per=FLAGS.ref_period)

    
   
    #COMPARISON WITH K-PAIRWISE AND DG MODELS (only for retinal data)
    if FLAGS.dataset=='retina':
        print('nearest sample analysis -----------------------------------')
        num_samples = 100 #this is for the 'nearest sample' analysis (Fig. S5)
        print('real_samples')
        analysis.nearest_sample(X_real=real_samples, fake_samples=real_samples, num_neurons=FLAGS.num_neurons, num_bins=FLAGS.num_bins, folder=FLAGS.sample_dir, name='real', num_samples=num_samples)
        ###################
        print('fake_samples')
        analysis.nearest_sample(X_real=real_samples, fake_samples=fake_samples.T, num_neurons=FLAGS.num_neurons, num_bins=FLAGS.num_bins, folder=FLAGS.sample_dir, name='spikeGAN', num_samples=num_samples)
        ###################
        k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=FLAGS.num_samples, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance, folder=os.getcwd()+'/data/retinal data/')    
        print('k_pairwise_samples')
        _,_,_,_ ,_ = analysis.get_stats(X=k_pairwise_samples, num_neurons=FLAGS.num_neurons, num_bins= FLAGS.num_bins, folder=FLAGS.sample_dir, name='k_pairwise', instance=FLAGS.data_instance)
        analysis.nearest_sample(X_real=real_samples, fake_samples=k_pairwise_samples, num_neurons=FLAGS.num_neurons, num_bins=FLAGS.num_bins, folder=FLAGS.sample_dir, name='k_pairwise', num_samples=num_samples)
        ###################
        DDG_samples = retinal_data.load_samples_from_DDG_model(num_samples=FLAGS.num_samples, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance, folder=os.getcwd()+'/data/retinal data/')    
        print('DDG_samples')
        _,_,_,_ ,_ = analysis.get_stats(X=DDG_samples, num_neurons=FLAGS.num_neurons, num_bins= FLAGS.num_bins, folder=FLAGS.sample_dir, name='DDG', instance=FLAGS.data_instance)
        analysis.nearest_sample(X_real=real_samples, fake_samples=DDG_samples, num_neurons=FLAGS.num_neurons, num_bins=FLAGS.num_bins, folder=FLAGS.sample_dir, name='DDG', num_samples=num_samples)
        
        
   
    
if __name__ == '__main__':
  tf.app.run()
