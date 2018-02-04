# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:25:22 2017

@author: manuel
"""

import numpy as np
import scipy.io as sio
from tflib import sim_pop_activity


def get_samples(num_bins=32, num_neurons=50, instance='1'):                        
    '''
    gets original retinal data (a matlab file)
    '''
    mat_contents = sio.loadmat('~/data/retinal_data/original_data_num_neurons_' + str(num_neurons) + '_' + instance + '.mat')   
    data = mat_contents['data']
    
    if num_bins!=1:
        X = np.zeros((num_bins*data.shape[1],data.shape[0]-num_bins))
        for ind_s in range(data.shape[0]-num_bins):
            sample = data[ind_s:ind_s+num_bins,:].T
            X[:,ind_s] = sample.reshape((num_neurons*num_bins,-1))[:,0] 
    else:
        X = data.T
    sim_pop_activity.plot_samples(X, num_neurons=num_neurons, folder='', name='retinal_samples')
    return X
        
        
def load_samples_from_k_pairwise_model(num_samples=2**13, num_bins=32, num_neurons=50, instance='1', folder='~/data/retinal data/'):
    '''
    gets data simulated by the k-pairwise method that approximates either the retinal data (Fig. 3) or the negative correlations data (Fig. S6)
    '''
    mat_contents = sio.loadmat(folder + '/k_pairwise_num_neurons_' + str(num_neurons) + '_' + instance + '.mat')    
    data = mat_contents['samples_batch_all']
    if num_bins!=1:
        data = data[0:data.shape[0]-data.shape[0]%num_bins,:]
        X = np.zeros((num_bins*data.shape[1],int(np.min([num_samples,data.shape[0]/num_bins]))))
        for ind_s in range(int(np.min([num_samples,data.shape[0]/num_bins]))):
            sample = data[ind_s*num_bins:(ind_s+1)*num_bins,:].T
            X[:,ind_s] = sample.reshape((num_neurons*num_bins,-1))[:,0] 
    
    else:
        X = data.T
        
    return X
   

def load_samples_from_DDG_model(num_samples=2**13, num_bins=32, num_neurons=50, instance='1', folder='~/data/retinal data/'):
    '''
    gets data simulated by the DG method that approximates either the retinal data (Fig. 3) or the negative correlations data (Fig. S6)
    '''
    mat_contents = sio.loadmat(folder + '/DG_num_neurons_' + str(num_neurons) + '_' + instance + '.mat')    
    data = mat_contents['data']
    if num_bins!=1:
        data = data[0:data.shape[0]-data.shape[0]%num_bins,:]
        X = np.zeros((num_bins*data.shape[1],int(np.min([num_samples,data.shape[0]/num_bins]))))
        for ind_s in range(int(np.min([num_samples,data.shape[0]/num_bins]))):
            sample = data[ind_s*num_bins:(ind_s+1)*num_bins,:].T
            X[:,ind_s] = sample.reshape((num_neurons*num_bins,-1))[:,0] 
    
    else:
        X = data.T
        
    return X
   

