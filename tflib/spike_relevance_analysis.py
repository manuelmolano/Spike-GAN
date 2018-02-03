#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:56:25 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:13:25 2017

@author: manuel
"""
import sys, os
print(os.getcwd())
sys.path.append('/home/manuel/improved_wgan_training/')
import tensorflow as tf

import os
import pprint
from model_conv import WGAN_conv
from tflib import sim_pop_activity, retinal_data, analysis#, visualize_filters_and_units, sim_pop_activity
import numpy as np
#from utils import pp, get_samples_autocorrelogram, get_samples
import matplotlib.pyplot as plt
import matplotlib
import time
#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


#parameters used for (some) figures
flags = tf.app.flags
flags.DEFINE_string("architecture", "conv", "semi-conv (conv) or fully connected (fc)")
flags.DEFINE_integer("num_iter", 300000, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam [1e-4]")
flags.DEFINE_float("beta1", 0., "Momentum term of adam [0.]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("training_step", 200, "number of batches between weigths and performance saving")
flags.DEFINE_string("training_stage", '', "stage of the training used for the GAN")
flags.DEFINE_integer("num_layers", 4, "number of convolutional layers [4]")
flags.DEFINE_integer("num_features", 4, "features in first layers [4]")
flags.DEFINE_integer("kernel_width", 4, "width of kernel [4]")

#parameter set specifiying data
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
flags.DEFINE_integer("group_size", 2, "size of the correlated groups (e.g. 2 means pairwise correlations)")
flags.DEFINE_integer("critic_iters", 5, "number of times the discriminator will be updated")
flags.DEFINE_float("lambd", 10, "parameter gradient penalization")
flags.DEFINE_integer("step", 2, "steps for the sliding window used to shuffle the spiketimes")
flags.DEFINE_integer("pattern_size", 8, "size of the sliding window")
flags.DEFINE_float("noise_in_packet", 0, "std of gaussian noise added to packets")
flags.DEFINE_integer("number_of_modes", 1, "[1,2] Number of different responses in the packet simulation. If =2 each type \
                     of packet will happen only once in the sample and one of two possible set of neurons will be chosen with equal prob for each packet.")
FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()
def main(_):
    #print parameters
    pp.pprint(flags.FLAGS.__flags)
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
    elif FLAGS.dataset=='packets':
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


    with tf.Session() as sess:
        wgan = WGAN_conv(sess,
        num_neurons=FLAGS.num_neurons,
        num_bins=FLAGS.num_bins,
        num_layers=FLAGS.num_layers,
        num_features=FLAGS.num_features,
        kernel_width=FLAGS.kernel_width,
        lambd=FLAGS.lambd,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)
        if not wgan.load(FLAGS.training_stage):
            raise Exception("[!] Train a model first, then run test mode")      
            
        num1 = 4
        num2 = 4
        num_samples = 8000
        if FLAGS.dataset=='retina':
            samples = retinal_data.get_samples(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance).T
        else:
            original_dataset = np.load(FLAGS.sample_dir+ '/stats_real.npz')
            if FLAGS.number_of_modes==1:
                _ = sim_pop_activity.spike_train_transient_packets(num_samples=num_samples, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size,\
                                                                     prob_packets=FLAGS.packet_prob,firing_rates_mat=original_dataset['firing_rate_mat'], refr_per=FLAGS.ref_period,\
                                                                     shuffled_index=original_dataset['shuffled_index'], limits=[0,64], groups=[0,1,2,3], folder=FLAGS.sample_dir, save_packet=True).T
                
                samples = sim_pop_activity.spike_train_transient_packets(num_samples=num_samples, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size,\
                                                                     prob_packets=0.2,firing_rates_mat=original_dataset['firing_rate_mat'], refr_per=FLAGS.ref_period,\
                                                                     shuffled_index=original_dataset['shuffled_index'], limits=[16,32], groups=[0], folder=FLAGS.sample_dir, save_packet=False).T
            elif FLAGS.number_of_modes==2:
                samples = original_dataset['samples'].T  
                num_samples = np.min([num_samples,samples.shape[0]])
                stim1_samples = np.zeros((int(num_samples/2),FLAGS.num_neurons,FLAGS.num_bins))                                                        
                stim2_samples = np.zeros((int(num_samples/2),FLAGS.num_neurons,FLAGS.num_bins))                                                        
                for ind_s in range(int(num_samples/2)):
                    stim1_samples[ind_s,:,:] ,_ = sim_pop_activity.spike_train_packets(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size, firing_rates_mat=original_dataset['firing_rate_mat'], \
                                                         refr_per=FLAGS.ref_period, noise=FLAGS.noise_in_packet, number_of_modes=FLAGS.number_of_modes, save_sample=True, folder=FLAGS.sample_dir, mode=0)
                    stim2_samples[ind_s,:,:],_ = sim_pop_activity.spike_train_packets(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size, firing_rates_mat=original_dataset['firing_rate_mat'], \
                                                         refr_per=FLAGS.ref_period, noise=FLAGS.noise_in_packet, number_of_modes=FLAGS.number_of_modes, save_sample=True, folder=FLAGS.sample_dir, mode=1)
        
                packets = {'packets_stim1':stim1_samples,'packets_stim2':stim2_samples}
                np.savez(FLAGS.sample_dir+'packets.npz',**packets)
                sim_pop_activity.spike_train_packets(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size, firing_rates_mat=original_dataset['firing_rate_mat'], \
                                                     refr_per=FLAGS.ref_period, noise=0, number_of_modes=FLAGS.number_of_modes, save_sample=True, folder=FLAGS.sample_dir, mode=0)
                    
                sim_pop_activity.spike_train_packets(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size, firing_rates_mat=original_dataset['firing_rate_mat'], \
                                                     refr_per=FLAGS.ref_period, noise=0, number_of_modes=FLAGS.number_of_modes, save_sample=True, folder=FLAGS.sample_dir, mode=1)
        
        inputs = tf.placeholder(tf.float32, name='inputs_to_discriminator', shape=[None, FLAGS.num_neurons*FLAGS.num_bins]) 
        score = wgan.get_critics_output(inputs)
        #real samples    
        f1,sbplt1 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        f2,sbplt2 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        f3,sbplt3 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
           
        step = FLAGS.step
        pattern_size = FLAGS.pattern_size
        times = step*np.arange(int(FLAGS.num_bins/step))
        times = np.delete(times,np.nonzero(times>FLAGS.num_bins-pattern_size))
        #print(times)
        importance_time_vector = np.zeros((num_samples,FLAGS.num_bins))
        importance_neuron_vector = np.zeros((num_samples,FLAGS.num_neurons))
        grad_maps = np.zeros((num_samples,FLAGS.num_neurons,FLAGS.num_bins))
        activity_map = np.zeros((FLAGS.num_neurons,FLAGS.num_bins))
        importance_time_vector_surr = np.zeros((num_samples,FLAGS.num_bins))
        importance_neuron_vector_surr = np.zeros((num_samples,FLAGS.num_neurons))
        sample_diff = np.zeros((FLAGS.num_neurons,FLAGS.num_bins))
        samples = samples[0:num_samples,:]
        for i in range(num_samples):
            sample = samples[i,:]
            time0 = time.time()
            grad_maps[i,:,:], _,sample_diff_aux = patterns_relevance(sample, FLAGS.num_neurons, score, inputs, sess, pattern_size, times)
            time1 = time.time()
            importance_time_vector[i,:] = np.mean(grad_maps[i,:,:],axis=0)#/max(np.mean(grads,axis=0))
            importance_neuron_vector[i,:]  = np.mean(grad_maps[i,:,:],axis=1)#/max(np.mean(grads,axis=1))
            sample_diff += sample_diff_aux
            
            #compute surrogate data 
            aux,_,_ = patterns_relevance(sample, FLAGS.num_neurons, score, inputs, sess, pattern_size, times, shuffle=True)
            time1 = time.time()
            importance_time_vector_surr[i,:] = np.mean(aux,axis=0)#/max(np.mean(grads,axis=0))
            importance_neuron_vector_surr[i,:]  = np.mean(aux,axis=1)#/max(np.mean(grads,axis=1))
            
            
            sample = sample.reshape(FLAGS.num_neurons,-1)
            activity_map += sample
            print(str(i) + ' time ' + str(time1 - time0))
            
        stimulus_id = np.load(FLAGS.sample_dir + '/stim.npz')['stimulus']
        
        stimulus_id = stimulus_id[0:num_samples]
        predicted_packets = analysis.find_packets(grad_maps,samples,FLAGS.num_neurons, FLAGS.num_bins, FLAGS.sample_dir, num_samples)
        ground_truth_packets_stim1 = np.mean(analysis.find_packets(stim1_samples,stim1_samples-1,FLAGS.num_neurons, FLAGS.num_bins, FLAGS.sample_dir, int(num_samples/2), plot_fig=False),axis=0)
        ground_truth_packets_stim2 = np.mean(analysis.find_packets(stim2_samples,stim2_samples-1,FLAGS.num_neurons, FLAGS.num_bins, FLAGS.sample_dir, int(num_samples/2), plot_fig=False),axis=0)
        
        importance_vectors = {'time':importance_time_vector, 'neurons':importance_neuron_vector, 'grad_maps':grad_maps, 'samples':samples, 'activity_map':activity_map,\
                              'time_surr':importance_time_vector_surr, 'neurons_surr':importance_neuron_vector_surr, 'sample_diff':sample_diff, 'predicted_packets':predicted_packets,\
                              'ground_truth_packets_stim1':ground_truth_packets_stim1, 'ground_truth_packets_stim2':ground_truth_packets_stim2}
        np.savez(FLAGS.sample_dir+'importance_vectors_'+str(step)+'_'+str(pattern_size)+'_'+str(num_samples)+'.npz',**importance_vectors)
 


def spikes_relevance(sample, wgan, sess):
    sample = sample.reshape((sample.shape[0],1))
    score = wgan.get_critics_output(np.concatenate((sample,sample),axis=1))[0].eval(session=sess)
    spikes = np.nonzero(sample)[0]
    grad = np.zeros((sample.shape[0],))
    for ind_spk in range(len(spikes)):
        aux_sample = sample.copy()
        aux_sample[spikes[ind_spk]] = 0
        aux = wgan.get_critics_output(np.concatenate((aux_sample,aux_sample),axis=1))[0].eval(session=sess) - score
        grad[spikes[ind_spk]] = aux
        
    return grad
        
def patterns_relevance(sample_original, num_neurons, score, inputs, sess, pattern_size, times, shuffle=False):
    #start_time = time.time()
    num_sh = 5
    dim = sample_original.shape[0]
    sample = sample_original.copy()
    sample[sample>1] = 1
    if shuffle:
       sample = sample.reshape((num_neurons,-1))
       for ind_2 in range(num_neurons):
           aux_pattern = sample[ind_2,:]
           np.random.shuffle(aux_pattern.T)
           
    sample = sample.reshape((1,dim))
    score_original = sess.run(score, feed_dict={inputs: np.concatenate((sample,sample),axis=0)})[0]
    sample = sample.reshape((num_neurons,-1))
    

    samples_shuffled = np.zeros((num_sh,num_neurons*times.shape[0],dim))
    sample_diff = np.zeros(sample.shape)
    for ind_sh in range(num_sh):
        counter = 0
        for ind_1 in range(times.shape[0]):
            for ind_2 in range(num_neurons):
                aux_sample = sample.copy()
                #you have to do this slicing like that, otherwise you will create a copy and the shuffling will not affect the original matrix
                aux_pattern = aux_sample[ind_2,times[ind_1]:times[ind_1]+pattern_size]
                np.random.shuffle(aux_pattern.T)
                sample_diff += aux_sample-sample
                samples_shuffled[ind_sh,counter,:] = aux_sample.flatten()
                counter += 1
        #grad_test[ind_sh,:] = np.abs(score - wgan.get_critics_output(samples_shuffled[ind_sh,:,:]).eval(session=sess))
        #aux = np.abs(score - wgan.get_critics_output(np.concatenate((samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],\
                                                                     #samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],\
                                                                     #samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:]),axis=0)).eval(session=sess))
        #aux2 = np.abs(score - wgan.get_critics_output(samples_shuffled[ind_sh,:,:]).eval(session=sess))
        #print(grad_test[ind_sh,:])
        #print(aux[0:4])
        #print(aux[4:8])
        #print('----')
        #assert np.all(aux[0:2]==aux2)
    #assert np.sum(np.std(samples_shuffled[0,:,:],axis=0))!=0
    
    aux = samples_shuffled.reshape((num_neurons*times.shape[0]*num_sh,dim))
    scores = sess.run(score, feed_dict={inputs: aux})
    grad = np.abs(score_original - scores)
    grad = grad.reshape((num_sh,num_neurons*times.shape[0]))
        
      
    grad = np.mean(grad,axis=0)
    grad_map = np.zeros(sample.shape)
    counting_map = np.zeros(sample.shape)
    counter = 0
    for ind_1 in range(times.shape[0]):
        for ind_2 in range(num_neurons):
            grad_map[ind_2,times[ind_1]:times[ind_1]+pattern_size] += grad[counter]*sample[ind_2,times[ind_1]:times[ind_1]+pattern_size]
            counting_map[ind_2,times[ind_1]:times[ind_1]+pattern_size] += sample[ind_2,times[ind_1]:times[ind_1]+pattern_size]
            counter += 1

            
    
    assert np.sum(grad_map[counting_map==0].flatten())==0
    grad_map /= counting_map
    grad_map[counting_map==0] = 0
    
#    print(np.unique(counting_map))
    return grad_map, grad, sample_diff 
    
if __name__ == '__main__':
  tf.app.run()
  



#[ 0.11054063  0.96706271  0.09767437  0.10173178]
#[ 0.11054277  0.9670608   0.09767365  0.10172868]
#[ 0.11054277  0.9670608   0.09767365  0.10172868]




