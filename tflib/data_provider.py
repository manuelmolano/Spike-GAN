#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:44:19 2017

@author: manuel
"""


import os
import numpy as np
from tflib import sim_pop_activity, retinal_data, analysis


def generate_spike_trains(config, recovery_dir):
    '''
    this function returns the training and dev sets, corresponding to the parameters provided in config
    '''
    if config.dataset=='uniform':
        if recovery_dir!="":
            aux = np.load(recovery_dir+ '/stats_real.npz')
            real_samples = aux['samples']
            firing_rates_mat = aux['firing_rate_mat']
            correlations_mat = aux['correlation_mat']
            shuffled_index = aux['shuffled_index']
        else:
            #shuffle neurons 
            shuffled_index = np.arange(config.num_neurons)
            np.random.shuffle(shuffled_index)
            firing_rates_mat = config.firing_rate+2*(np.random.random(int(config.num_neurons/config.group_size),)-0.5)*config.firing_rate/2    
            correlations_mat = config.correlation+2*(np.random.random(int(config.num_neurons/config.group_size),)-0.5)*config.correlation/2   
            #peaks of activity
            aux = np.arange(int(config.num_neurons/config.group_size))
            #peak of activity equal for all neurons 
            real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=config.num_bins,\
                                num_neurons=config.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, shuffled_index=shuffled_index,\
                                refr_per=config.ref_period,firing_rates_mat=firing_rates_mat, folder=config.sample_dir)
            
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, shuffled_index=shuffled_index,\
                           name='real',firing_rate_mat=firing_rates_mat, correlation_mat=correlations_mat)
            
        #get dev samples
        dev_samples = sim_pop_activity.get_samples(num_samples=int(config.num_samples/4), num_bins=config.num_bins,\
                       num_neurons=config.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, shuffled_index=shuffled_index,\
                       refr_per=config.ref_period,firing_rates_mat=firing_rates_mat)
        
    elif config.dataset=='packets':
        if recovery_dir!="":
            aux = np.load(recovery_dir+ '/stats_real.npz')
            real_samples = aux['samples']
            firing_rates_mat = aux['firing_rate_mat']
            shuffled_index = aux['shuffled_index']
        else:
            #shuffle the neurons 
            shuffled_index = np.arange(config.num_neurons)
            np.random.shuffle(shuffled_index)
            firing_rates_mat = config.firing_rate+2*(np.random.random(size=(config.num_neurons,1))-0.5)*config.firing_rate/2 
            real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=config.num_bins, refr_per=config.ref_period,\
                                 num_neurons=config.num_neurons, group_size=config.group_size, firing_rates_mat=firing_rates_mat, packets_on=True,\
                                 prob_packets=config.packet_prob, shuffled_index=shuffled_index, folder=config.sample_dir, noise_in_packet=config.noise_in_packet, number_of_modes=config.number_of_modes)
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, name='real',\
                       firing_rate_mat=firing_rates_mat, shuffled_index=shuffled_index)
        #get dev samples
        dev_samples = sim_pop_activity.get_samples(num_samples=int(config.num_samples/4), num_bins=config.num_bins, refr_per=config.ref_period,\
                       num_neurons=config.num_neurons, group_size=config.group_size, firing_rates_mat=firing_rates_mat, packets_on=True,\
                       prob_packets=config.packet_prob, shuffled_index=shuffled_index, noise_in_packet=config.noise_in_packet, number_of_modes=config.number_of_modes)
        
    elif config.dataset=='retina':
        dirpath = os.getcwd()
        real_samples = retinal_data.get_samples(num_bins=config.num_bins, num_neurons=config.num_neurons, instance=config.data_instance, folder=dirpath+'/data/retinal data/')
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, name='real',instance=config.data_instance)
        dev_samples = []
    return real_samples, dev_samples





