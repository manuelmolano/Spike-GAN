# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:41:38 2017

@author: manuel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from tflib import sim_pop_activity, retinal_data, figures#, data_provider
import time
#from sklearn.cluster import KMeans
import itertools
import seaborn as sns
#parameters for figures
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


def get_stats(X, num_neurons, num_bins, folder, name, firing_rate_mat=[],correlation_mat=[], activity_peaks=[], critic_cost=np.nan, instance='1',shuffled_index=[]): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    X_binnarized = (X > np.random.random(X.shape)).astype(float)   
    resave_real_data = False
    if name!='real':
        original_data = np.load(folder + '/stats_real.npz')   
        if any(k not in original_data for k in ("mean","acf","cov_mat","k_probs","lag_cov_mat","firing_average_time_course")):
            if 'samples' not in original_data:
                samples = retinal_data.get_samples(num_bins=num_bins, num_neurons=num_neurons, instance=instance)
            else:
                samples = original_data['samples']
            cov_mat_real, k_probs_real, mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, lag_cov_mat_real =\
            get_stats_aux(samples, num_neurons, num_bins)
            assert np.all(autocorrelogram_mat_real==original_data['acf'])
            assert np.all(mean_spike_count_real==original_data['mean'])       
            resave_real_data = True
        else:
            mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
            [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
    
    cov_mat, k_probs, mean_spike_count, autocorrelogram_mat, firing_average_time_course, lag_cov_mat = get_stats_aux(X_binnarized, num_neurons, num_bins)
    variances = np.diag(cov_mat)
    only_cov_mat = cov_mat.copy()
    only_cov_mat[np.diag_indices(num_neurons)] = np.nan
    
    #PLOT
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    f,sbplt = plt.subplots(2,3,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    #plot autocorrelogram(s)
    sbplt[1][1].plot(index, autocorrelogram_mat,'r')
    if name!='real':
        sbplt[1][1].plot(index, autocorrelogram_mat_real,'b')
        acf_error = np.sum(np.abs(autocorrelogram_mat-autocorrelogram_mat_real))
    sbplt[1][1].set_title('Autocorrelogram')
    sbplt[1][1].set_xlabel('time (ms)')
    sbplt[1][1].set_ylabel('number of spikes')
    
    #plot mean firing rates
    if name!='real':
        sbplt[0][0].plot([0,np.max(mean_spike_count_real)],[0,np.max(mean_spike_count_real)],'k')
        sbplt[0][0].plot(mean_spike_count_real,mean_spike_count,'.g')
        mean_error = np.sum(np.abs(mean_spike_count-mean_spike_count_real))
        sbplt[0][0].set_xlabel('mean firing rate expt')
        sbplt[0][0].set_ylabel('mean firing rate model')
    else:
        sbplt[0][0].plot(mean_spike_count,'b')
        sbplt[0][0].set_xlabel('neuron')
        sbplt[0][0].set_ylabel('firing probability')
        
    sbplt[0][0].set_title('mean firing rates')

    #plot covariances
    if name!='real':
        variances_real = np.diag(cov_mat_real)
        only_cov_mat_real = cov_mat_real.copy()
        only_cov_mat_real[np.diag_indices(num_neurons)] = np.nan
        sbplt[0][1].plot([np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],\
                        [np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],'k')
        sbplt[0][1].plot(only_cov_mat_real.flatten(),only_cov_mat.flatten(),'.g')
        sbplt[0][1].set_title('pairwise covariances')
        sbplt[0][1].set_xlabel('covariances expt')
        sbplt[0][1].set_ylabel('covariances model')
        corr_error = np.nansum(np.abs(only_cov_mat-only_cov_mat_real).flatten())
    else:       
        map_aux = sbplt[0][1].imshow(only_cov_mat,interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[0][1])
        sbplt[0][1].set_title('covariance mat')
        sbplt[0][1].set_xlabel('neuron')
        sbplt[0][1].set_ylabel('neuron')
        
        
    #plot k-statistics
    if name!='real':
        sbplt[1][0].plot([0,np.max(k_probs_real)],[0,np.max(k_probs_real)],'k')        
        sbplt[1][0].plot(k_probs_real,k_probs,'.g')        
        k_probs_error = np.sum(np.abs(k_probs-k_probs_real))
        sbplt[1][0].set_xlabel('k-probs expt')
        sbplt[1][0].set_ylabel('k-probs model')
    else:
        sbplt[1][0].plot(k_probs)
        sbplt[1][0].set_xlabel('K')
        sbplt[1][0].set_ylabel('probability')
        
    sbplt[1][0].set_title('k statistics')        
      
    #plot average time course
    #firing_average_time_course[firing_average_time_course>0.048] = 0.048
    map_aux = sbplt[0][2].imshow(firing_average_time_course,interpolation='nearest')
    f.colorbar(map_aux,ax=sbplt[0][2])
    sbplt[0][2].set_title('sim firing time course')
    sbplt[0][2].set_xlabel('time (ms)')
    sbplt[0][2].set_ylabel('neuron')
    if name!='real':
        map_aux = sbplt[1][2].imshow(firing_average_time_course_real,interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[1][2])
        sbplt[1][2].set_title('real firing time course')
        sbplt[1][2].set_xlabel('time (ms)')
        sbplt[1][2].set_ylabel('neuron')
        time_course_error = np.sum(np.abs(firing_average_time_course-firing_average_time_course_real).flatten())    
    
    f.savefig(folder+'stats_'+name+'_II.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    
    if name!='real':   
        #PLOT LAG COVARIANCES
        #figure for all training error across epochs (supp. figure 2)
        f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)    
        map_aux = sbplt[0][0].imshow(lag_cov_mat_real,interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[0][0])
        sbplt[0][0].set_title('lag covariance mat expt')
        sbplt[0][0].set_xlabel('neuron')
        sbplt[0][0].set_ylabel('neuron shifted')
        map_aux = sbplt[1][0].imshow(lag_cov_mat,interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[1][0])
        sbplt[1][0].set_title('lag covariance mat model')
        sbplt[1][0].set_xlabel('neuron')
        sbplt[1][0].set_ylabel('neuron shifted')
        lag_corr_error = np.nansum(np.abs(lag_cov_mat-lag_cov_mat_real).flatten())
        sbplt[0][1].plot([np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],\
                        [np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],'k')        
        sbplt[0][1].plot(lag_cov_mat_real,lag_cov_mat,'.g')
        sbplt[0][1].set_xlabel('lag cov real')
        sbplt[0][1].set_ylabel('lag cov model')
        sbplt[1][1].plot([np.min(variances_real.flatten()),np.max(variances.flatten())],\
                        [np.min(variances_real.flatten()),np.max(variances_real.flatten())],'k')
        sbplt[1][1].plot(variances_real.flatten(),variances.flatten(),'.g')
        sbplt[1][1].set_title('variances')
        sbplt[1][1].set_xlabel('variances expt')
        sbplt[1][1].set_ylabel('variances model')
        variance_error = np.nansum(np.abs(variances_real-variances).flatten())
        f.savefig(folder+'lag_covs_'+name+'_II.svg',dpi=600, bbox_inches='tight')
        plt.close(f)
        
    if name=='real' and len(firing_rate_mat)>0:
        #ground truth data but not real (retinal) data
        data = {'mean':mean_spike_count, 'acf':autocorrelogram_mat, 'cov_mat':cov_mat, 'samples':X, 'k_probs':k_probs,'lag_cov_mat':lag_cov_mat,\
        'firing_rate_mat':firing_rate_mat, 'correlation_mat':correlation_mat, 'activity_peaks':activity_peaks, 'shuffled_index':shuffled_index, 'firing_average_time_course':firing_average_time_course}
        np.savez(folder + '/stats_'+name+'.npz', **data)    
    else:
        data = {'mean':mean_spike_count, 'acf':autocorrelogram_mat, 'cov_mat':cov_mat, 'k_probs':k_probs, 'firing_average_time_course':firing_average_time_course,\
                'critic_cost':critic_cost, 'lag_cov_mat':lag_cov_mat}
        np.savez(folder + '/stats_'+name+'.npz', **data)   
        if resave_real_data:
            if 'firing_rate_mat' in original_data:
                data = {'mean':mean_spike_count_real, 'acf':autocorrelogram_mat_real, 'cov_mat':cov_mat_real, 'samples':samples, 'k_probs':k_probs_real,'lag_cov_mat':lag_cov_mat_real,\
                'firing_rate_mat':original_data['firing_rate_mat'], 'correlation_mat':original_data['correlation_mat'], 'activity_peaks':original_data['activity_peaks'],\
                 'shuffled_index':original_data['shuffled_index'], 'firing_average_time_course':firing_average_time_course_real}
            else:
                data = {'mean':mean_spike_count_real, 'acf':autocorrelogram_mat_real, 'cov_mat':cov_mat_real, 'samples':samples, 'k_probs':k_probs_real,'lag_cov_mat':lag_cov_mat_real,\
                    'firing_average_time_course':firing_average_time_course_real}
            np.savez(folder + '/stats_real.npz', **data)     
        if name!='real': 
            errors_mat = {'acf_error':acf_error, 'mean_error':mean_error, 'corr_error':corr_error, 'time_course_error':time_course_error, 'k_probs_error':k_probs_error,\
                          'variance_error':variance_error, 'lag_corr_error':lag_corr_error}
            np.savez(folder + '/errors_'+name+'.npz', **errors_mat)
            samples_fake = {'samples':X}
            np.savez(folder + '/samples_'+name[0:4]+'.npz', **samples_fake)
            return acf_error, mean_error, corr_error, time_course_error, k_probs_error
    

def get_stats_aux(X, num_neurons, num_bins):
    '''
    auxiliary function of get_stats, computes the covariance, k-probabilities, average firing rates, autocorrelogram and lag-covariances
    '''
    lag = 10
    num_samples = X.shape[1]
    spike_count = np.zeros((num_neurons,num_samples))
    X_continuous = np.zeros((num_neurons,num_bins*num_samples))
    autocorrelogram_mat = np.zeros(2*lag+1)
    firing_average_time_course = np.zeros((num_neurons,num_bins))
   
    for ind in range(num_samples):
        sample = X[:,ind].reshape((num_neurons,-1))
        spike_count[:,ind] = np.sum(sample,axis=1)        
        X_continuous[:,ind*num_bins:(ind+1)*num_bins] = sample
        autocorrelogram_mat += autocorrelogram(sample,lag=lag)
        firing_average_time_course += sample
  
    #covariance mat
    cov_mat =  np.cov(X_continuous)
    #k-probs
    aux = np.histogram(np.sum(X_continuous,axis=0),bins=np.arange(num_neurons+2)-0.5)[0]
    k_probs = aux/X_continuous.shape[1]   
    #average firing rate
    mean_spike_count = np.mean(spike_count,axis=1)
    #autocorrelogram
    autocorrelogram_mat = autocorrelogram_mat/np.max(autocorrelogram_mat)
    autocorrelogram_mat[lag] = 0
    #average time course
    firing_average_time_course = firing_average_time_course/num_samples
    #lag cov mat
    lag_cov_mat = np.zeros((num_neurons,num_neurons))
    for ind_n in range(num_neurons):
        for ind_n2 in range(num_neurons):
            resp1 = X_continuous[ind_n,0:num_bins*num_samples-1].reshape(1,-1)
            resp2 = X_continuous[ind_n2,1:num_bins*num_samples].reshape(1,-1)
            aux = np.cov(np.concatenate((resp1,resp2),axis=0))
            lag_cov_mat[ind_n,ind_n2] = aux[1,0]
            
    return cov_mat, k_probs, mean_spike_count, autocorrelogram_mat, firing_average_time_course, lag_cov_mat



def triplet_corr(X, num_neurons, num_bins, folder, name, set_size=3): 
    '''
    computes correlations among triplets of neurons
    
    '''
    num_samples = X.shape[1]
    X_binnarized = (X > np.random.random(X.shape)).astype(float)   
       
    
    X_binnarized = np.reshape(X_binnarized,newshape=(num_neurons,num_bins,num_samples))
    X_binnarized = np.transpose(X_binnarized,axes=(0,2,1))
    X_binnarized = np.reshape(X_binnarized,newshape=(num_neurons,num_bins*num_samples))
    
    
    X_binnarized -= np.mean(X_binnarized,axis=1).reshape((num_neurons,1))
    
    assert np.all(np.abs(np.mean(X_binnarized,axis=1))<0.0000000000001)
    
    if name!='real':
        corrs_real = np.load(folder + '/triplet_corr_real.npz')     

    
    combinations = list(itertools.combinations(range(num_neurons),set_size))
    tr_corrs = np.zeros((len(combinations),1))
    for ind_comb in range(len(combinations)):
        if ind_comb%100==0:
            time0 = time.time()
        prod = 1
        for ind in range(set_size):
            prod *= X_binnarized[combinations[ind_comb][ind],:]
        tr_corrs[ind_comb] = np.mean(prod)
        if ind_comb%100==0:
            print(name + ' ' + str(ind_comb) + ' out of ' + str(len(combinations)) + ' time ' + str(time.time() - time0))
    plt.hist(tr_corrs)
    data = {'tr_corrs':tr_corrs}
    np.savez(folder + '/triplet_corr_'+name+'.npz', **data)
    
    if name!='real':
        f = plt.figure(figsize=(8, 8),dpi=250)
        maximo = np.max(np.array([np.max(corrs_real['tr_corrs']),np.max(tr_corrs)]))
        minimo = np.min(np.array([np.min(corrs_real['tr_corrs']),np.min(tr_corrs)]))
        plt.plot([minimo,maximo],[minimo,maximo],'k')
        plt.plot(corrs_real['tr_corrs'],tr_corrs,'.')
        f.savefig(folder + '/triplet_corr_'+name+'.svg',dpi=600, bbox_inches='tight')
        plt.close(f)
    
    
    
def evaluate_approx_distribution(X, folder, num_samples_theoretical_distr=2**15, num_bins=10, num_neurons=4, group_size=2, refr_per=2): 
    '''
    compute numerical probabilities from ground truth, surrogate and generated dataset (see Fig. S1)
    
    '''
    num_samples = X.shape[1]
    #get freqs of real samples
    original_data = np.load(folder + '/stats_real.npz')        
    real_samples = original_data['samples']#[:,0:num_samples]
    #X = X[:,0:num_samples]
    X_binnarized = (X > np.random.random(X.shape)).astype(float)   
    if os.path.exists(folder+'/probs_ns_' + str(num_samples) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz'):
        probs = np.load(folder+'/probs_ns_' + str(num_samples) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz')
        sim_samples_freqs = probs['sim_samples_freqs'] #frequencies of each different pattern contained in the generated dataset       
        numerical_prob = probs['numerical_prob']#numerical probs of each different pattern contained in the generated dataset       
        freq_in_training_dataset = probs['freq_in_training_dataset']#frequencies of each different pattern in the generated dataset wrt the ground truth dataset
        num_impossible_samples = probs['num_impossible_samples'] #number of patterns with numerical prob = 0
        surr_samples_freqs = probs['surr_samples_freqs']#frequencies of each different pattern contained in all surrogate datasets (num surrogates = 100 so the vector sums up to 100)
        freq_in_training_dataset_surrogates = probs['freq_in_training_dataset_surrogates']#frequencies of each different pattern in in all surrogate datasets wrt the ground truth dataset
        numerical_prob_surrogates = probs['numerical_prob_surrogates']#numerical probs of each different pattern contained in the surrogate datasets       
        num_impossible_samples_surrogates = probs['num_impossible_samples_surrogates']
        num_probs = np.load(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz')        
        num_probs = num_probs['num_probs']
        theoretical_probs = num_probs[1]/np.sum(num_probs[1])
    else:
        #get numerical probabilities
        if os.path.exists(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz'):
            num_probs = np.load(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz')        
            num_probs = num_probs['num_probs']
        else:
            num_probs = sim_pop_activity.get_aproximate_probs(num_samples=num_samples_theoretical_distr,num_bins=num_bins, num_neurons=num_neurons, correlations_mat=original_data['correlation_mat'],\
                            group_size=group_size,refr_per=refr_per,firing_rates_mat=original_data['firing_rate_mat'], activity_peaks=original_data['activity_peaks'])
            numerical_probs = {'num_probs':num_probs}
            np.savez(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz',**numerical_probs)
        
        #samples
        samples_theoretical_probs = num_probs[0]
        #probabilites obtain from a large dataset    
        theoretical_probs = num_probs[1]/np.sum(num_probs[1])
        #get the freq of simulated samples in the original dataset, in the ground truth dataset and in the simulated dataset itself
        print('compute probs of generated samples')
        freq_in_training_dataset, numerical_prob, sim_samples_freqs = comparison_to_original_and_gt_datasets(samples=X_binnarized, real_samples=real_samples,\
                ground_truth_samples=samples_theoretical_probs, ground_truth_probs=theoretical_probs)
        #'impossible' samples: samples for which the theoretical prob is 0
        num_impossible_samples = np.count_nonzero(numerical_prob==0)
        #we will now perform the same calculation for several datasets extracted from the ground truth distribution   
        print('compute probs for surrogates')
        num_surr = 1
        freq_in_training_dataset_surrogates = np.zeros((num_surr*num_samples,)) 
        numerical_prob_surrogates = np.zeros((num_surr*num_samples,))
        surr_samples_freqs = np.zeros((num_surr*num_samples,))
        num_impossible_samples_surrogates =  np.zeros((num_surr, ))
        counter = 0
        for ind_surr in range(num_surr):
            if ind_surr%10==0:
                print(ind_surr)
            if ind_surr==num_surr:
                #as a control, the last surrogate is the original dataset itself
                surrogate= real_samples
                np.random.shuffle(surrogate.T)
                surrogate = surrogate[:,0:np.min((num_samples,surrogate.shape[1]))]
            else:
                surrogate = sim_pop_activity.get_samples(num_samples=num_samples, num_bins=num_bins,\
                    num_neurons=num_neurons, correlations_mat=original_data['correlation_mat'], group_size=group_size, refr_per=refr_per,\
                    firing_rates_mat=original_data['firing_rate_mat'], activity_peaks=original_data['activity_peaks'],shuffled_index=original_data['shuffled_index'])
                
            freq_in_training_dataset_aux, numerical_prob_aux, samples_freqs_aux = comparison_to_original_and_gt_datasets(samples=surrogate, real_samples=real_samples,\
                ground_truth_samples=samples_theoretical_probs, ground_truth_probs=theoretical_probs)
            
            if ind_surr==num_surr:
                assert all(freq_in_training_dataset_aux!=0)
            else:
                freq_in_training_dataset_surrogates[counter:counter+len(freq_in_training_dataset_aux)] = freq_in_training_dataset_aux
                numerical_prob_surrogates[counter:counter+len(freq_in_training_dataset_aux)] = numerical_prob_aux
                surr_samples_freqs[counter:counter+len(freq_in_training_dataset_aux)] = samples_freqs_aux
                num_impossible_samples_surrogates[ind_surr] = np.count_nonzero(numerical_prob_aux==0)
                counter += len(freq_in_training_dataset_aux)
        
        freq_in_training_dataset_surrogates = freq_in_training_dataset_surrogates[0:counter]
        numerical_prob_surrogates = numerical_prob_surrogates[0:counter]
        surr_samples_freqs = surr_samples_freqs[0:counter]
        probs = {'sim_samples_freqs':sim_samples_freqs, 'freq_in_training_dataset':freq_in_training_dataset, 'numerical_prob':numerical_prob, 'num_impossible_samples':num_impossible_samples,\
                'surr_samples_freqs':surr_samples_freqs, 'freq_in_training_dataset_surrogates':freq_in_training_dataset_surrogates, 'numerical_prob_surrogates': numerical_prob_surrogates,\
                'num_impossible_samples_surrogates': num_impossible_samples_surrogates}
        
        np.savez(folder+'/probs_ns_' + str(num_samples) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz',**probs)
        
    # load surrogate data
    data = probs #np.load('../../../data/probs_ns_8000_ns_gt_2097152.npz')
    freq_in_training_dataset_surrogates = data['freq_in_training_dataset_surrogates']
    numerical_prob_surrogates = data['numerical_prob_surrogates']
    num_impossible_samples_surrogates = data['num_impossible_samples_surrogates']
    surr_samples_freqs = data['surr_samples_freqs']
    num_impossible_samples = data['num_impossible_samples']
    
  
    
    # preprocess data
    # theoretical probabilities of "possible" samples in both the original and the generated data.
    # Note that all empirical frequencies and theoretical probabilities are always reported for the generated set (or the surrogate set below). In other words, there are no zeroes in sim_samples_freqs or in surr_samples_freqs.
    possible_in_gen = (data['numerical_prob']>0)# & (data['freq_in_training_dataset']>0)
    prob_log_possible_in_gen = np.log10(data['numerical_prob'][possible_in_gen])
    
    possible_in_surr = (data['numerical_prob_surrogates']>0)# & (data['freq_in_training_dataset']>0)
    prob_log_possible_in_surr = np.log10(data['numerical_prob_surrogates'][possible_in_surr])
    
    #empirical probabilities of "possible" samples in the generated data, wrt the generated data
    freq_log_possible_in_gen_wrt_gen = np.log10(data['sim_samples_freqs'][possible_in_gen])
    
    #empirical probabilities of "possible" samples in the surr data, wrt the surr data
    freq_log_possible_in_surr_wrt_surr = np.log10(data['surr_samples_freqs'][possible_in_surr])
            
    #figure
    print('plotting probs')
    fig, ax = plt.subplots(figsize=(5,5), nrows=1, ncols=1)
    #ax = ax.reshape((-1,1))
    
    sns.kdeplot(data=freq_log_possible_in_gen_wrt_gen,
                data2=prob_log_possible_in_gen,
                ax=ax, n_levels=10, bw=(0.1,0.1), shade=True, cmap="inferno")
    ax.plot([-6.65,-4.5], [-6.65,-4.5], color='#d8dcd6', linewidth=1)
    sns.kdeplot(data=freq_log_possible_in_surr_wrt_surr,
                data2=prob_log_possible_in_surr,
                ax=ax, n_levels=10, bw=(0.1,0.1), shade=False, cmap='Blues_r')
   
    ax.set_xlim((-6.6,-4.5))
    ax.set_ylim((-6.6,-4.5))
    ax.set_ylabel('log numerical probability',fontsize=10)
    ax.set_xlabel('log probability in surrogate and generated distr.',fontsize=10)

        
    plt.tight_layout()
    fig.savefig(folder+'empirical_theoretical_probs.pdf')
    print('num impossible samples surr')
    print(num_impossible_samples_surrogates)
    print('num impossible samples sim')
    print(num_impossible_samples)
    print('entropy ground truth')
    print(-np.sum(theoretical_probs*np.log2(theoretical_probs)))
    print('entropy surr')
    print(-np.sum(data['surr_samples_freqs']*np.log2(data['surr_samples_freqs'])))
    print('entropy sim')
    print(-np.sum(data['sim_samples_freqs']*np.log2(data['sim_samples_freqs'])))
    
    print('portion of surr samples that were present in the training dataset')
    print(np.sum((data['surr_samples_freqs'][data['freq_in_training_dataset_surrogates']>0])))
    print('portion of gen samples that were present in the training dataset')
    print(np.sum((data['sim_samples_freqs'][data['freq_in_training_dataset']>0])))
        
        
    

def comparison_to_original_and_gt_datasets(samples, real_samples, ground_truth_samples, ground_truth_probs):
    '''
    auxiliary function for evaluate_approx_distribution that computes the prob in the training data set, in the ground truth dataset and in the generated dataset
    '''
    #get freqs of simulated samples
    aux = np.unique(samples,axis=1,return_counts=True)
    sim_samples_probs = aux[1]/np.sum(aux[1])
    sim_samples_unique = aux[0]    
    print(sim_samples_unique.shape)
    #get freqs of original samples
    aux = np.unique(real_samples,axis=1,return_counts=True)
    original_samples_probs = aux[1]/np.sum(aux[1])
    original_samples = aux[0]
    #simulated samples that are not in the original dataset
    #if zero, the simulated sample is not present in the original dataset; 
    #if different from zero it stores the frequency with which the sample occurs in the original dataset
    prob_in_training_dataset = np.zeros((sim_samples_unique.shape[1],)) 
    #generated samples that are not in the ground truth dataset and thus have theoretical prob = 0
    #if zero, the simulated sample is not present in the ground truth dataset; 
    #if different from zero it stores the frequency with which the sample occurs 
    numerical_prob = np.zeros((sim_samples_unique.shape[1],))
    start_time = time.time()
    for ind_s in range(sim_samples_unique.shape[1]):
        if ind_s%1000==0:
            print(str(ind_s) + ' time ' + str(time.time() - start_time))
        #get sample
        sample = sim_samples_unique[:,ind_s].reshape(sim_samples_unique.shape[0],1)
        #check whether the sample is in the ground truth dataset and if so get prob
        looking_for_sample = np.equal(ground_truth_samples.T,sample.T).all(1)
        if any(looking_for_sample):
            numerical_prob[ind_s] = ground_truth_probs[looking_for_sample]
        
        #check whether the sample is in the original dataset and if so get prob
        looking_for_sample = np.equal(original_samples.T,sample.T).all(1)
        if any(looking_for_sample):
            prob_in_training_dataset[ind_s] = original_samples_probs[looking_for_sample]
       
            
    return prob_in_training_dataset, numerical_prob, sim_samples_probs        
    
def autocorrelogram(r,lag):
    '''
    computes the autocorrelogram
    '''
    #get autocorrelogram
    margin = np.zeros((r.shape[0],lag))
    #concatenate margins to then flatten the trials matrix
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    ac = np.zeros(2*lag+1)
    for ind_spk in range(len(spiketimes[0])):
        spike = spiketimes[0][ind_spk]
        ac = ac + r_flat[spike-lag:spike+lag+1]
        
    return ac    
 
    
 
def compare_GANs(folder, name, variables_compared):
    '''
    compares the results from different Spike-GAN architectures
    '''
    variables = {}
    
    folders = glob.glob(folder+name)
    num_rows = 1
    num_cols = 1
    
    leyenda = []
    variables = np.zeros((len(folders),len(variables_compared)))
    all_errors = np.zeros((len(folders),7))
    #go over all folders
    for ind_f in range(len(folders)):
        experiment = ''
        for ind in range(len(variables_compared)):
            #save parameter value
            variables[ind_f,ind] = float(find_value(folders[ind_f], variables_compared[ind]))
            experiment +=  str(variables[ind_f,ind]) + '  '
        leyenda.append(experiment)
        if os.path.exists(folders[ind_f]+'/errors_fake.npz'):
            errors = np.load(folders[ind_f]+'/errors_fake.npz')
        else:
            files = glob.glob(folders[ind_f]+'/errors_fake*.npz')
            latest_file = 'errors_fake'+str(find_latest_file(files,'errors_fake'))+'.npz'
            errors = np.load(folders[ind_f]+'/'+latest_file)
            
         
        errors_keys = errors.keys()  
        errors_keys.sort()
        counter = 0
        for key in errors_keys:
            all_errors[ind_f,counter] = errors[key]
            counter += 1
        

    #put together different interations of same experiment
    all_errors_mean, all_errors_std, unique_param, leyenda = merge_iterations(all_errors, variables, leyenda)
    
    maximos = np.max(all_errors_mean,axis=0).reshape((1,all_errors_mean.shape[1]))
    all_errors_mean = np.divide(all_errors_mean,maximos)
    mean_for_each_exp = np.mean(all_errors_mean,axis=1).reshape((all_errors_mean.shape[0],1))
    std_for_each_exp = np.std(all_errors_mean,axis=1).reshape((all_errors_mean.shape[0],1))
    all_errors_mean = np.concatenate((all_errors_mean,mean_for_each_exp),axis=1)
    all_errors_std = np.concatenate((all_errors_std,std_for_each_exp),axis=1)
    f,sbplt = plt.subplots(num_rows,num_cols,figsize=(10, 8),dpi=250)    
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for ind_exp in range(all_errors_mean.shape[0]):
        
        sbplt.errorbar(np.arange(8), all_errors_mean[ind_exp,:], yerr=all_errors_std[ind_exp,:])
        
    sbplt.legend(leyenda,loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
    errors_keys = list(errors_keys)
    errors_keys.append('mean')
    plt.setp(sbplt, xticks=np.arange(8), xticklabels=errors_keys)
    sbplt.set_xlim([-1,8])

def find_value(string, variable):
    '''
    finds a value in a string of a network parameter (e.g. num_layers)
    '''
    index1 = string.find(variable)+len(variable)+1
    if index1==len(variable):
        return np.nan
    aux = string[index1:]
    index2 = aux.find('_')
    
    if index2==-1:
        value = aux
    else:
        value = aux[0:index2]
        
    return value

def find_latest_file(files,name):
    '''
    if the training has not finished, it will find the file corresponding to the latest training stage
    '''
    maximo = 0
    for ind in range(len(files)):
        file = files[ind]
        aux = file[file.find(name)+len(name):file.find('npz')-1]
        maximo = np.max([float(aux),maximo])
        
    return str(int(maximo))


def merge_iterations(mat,parameters,leyenda):
    '''
    merges rows in mat that have equal parameters
    '''
    unique_param = np.unique(parameters, axis=0)
    mean_mat = np.zeros((unique_param.shape[0],mat.shape[1]))
    std_mat = np.zeros((unique_param.shape[0],mat.shape[1]))
    leyenda_red = []
    for ind_p in range(unique_param.shape[0]):
        index = np.sum(parameters==unique_param[ind_p,:],axis=1)==3
        mean_mat[ind_p,:] = np.mean(mat[index,:], axis=0)
        std_mat[ind_p,:] = np.std(mat[index,:], axis=0)
        leyenda_red.append(leyenda[np.nonzero(index)[0][0]])
    return mean_mat, std_mat, unique_param, leyenda_red


def compute_num_variables(num_bins=256, num_neurons=32, num_features=128, kernel=5, num_units=490):
    '''
    computes num of variables for a given architecture (only for two layers)
    '''
    num_layers = 2
    num_features *= 2
    print('conv')
    print(((num_neurons*num_features*kernel + num_features) + (num_features*2*num_features*kernel + 2*num_features) + (2*num_features*num_bins/num_layers**2) + 1) +\
          ((128*2*num_features*num_bins/num_layers**2 + 2*num_features*num_bins/num_layers**2) + (num_features*2*num_features*kernel + num_features) + (num_neurons*num_features*kernel + num_neurons)))

    print('fc')
    print(((num_neurons*num_bins*num_units) + 3*(num_units**2) + 4*(num_units) + num_units + 1) + \
          ((128*num_units) + 3*(num_units**2) + 4*(num_units) + (num_units*num_bins*num_neurons) + (num_bins*num_neurons)))
    
 
    
def get_predicted_packets(folder,threshold=95):
    '''
    estimates the original packets associated with the dataset in folder
    '''
    plt.close('all')
    real_data = np.load(folder + '/stats_real.npz')
    stimulus_id = np.load(folder + '/stim.npz')['stimulus']
    importance_info = np.load(folder + '/importance_vectors_1_8_8000.npz')
    grads = importance_info['grad_maps']
    num_samples = grads.shape[0]
    num_neurons = grads.shape[1]
    num_bins = grads.shape[2]
    samples = importance_info['samples']
    
    stimulus_id = stimulus_id[0:num_samples]
    find_packets(real_data,grads,samples,num_neurons, num_bins, folder, num_samples, stimulus_id, threshold_prct=threshold)
    
def find_packets(grad_maps,samples,num_neurons, num_bins, folder, num_samples, threshold_prct=95, plot_fig=True):
    '''
    auxiliary function of get_predicted_packets. Thresholds, aligns and averages importance maps to estimate the original activity packets (Fig. S2)
    '''
    if plot_fig:
        index = np.arange(num_neurons)#np.argsort(original_dataset['shuffled_index'])#
        sim_pop_activity.plot_samples(grad_maps.reshape((grad_maps.shape[0],grad_maps.shape[1]*grad_maps.shape[2])).T, num_neurons, folder, 'grad_maps', index=index)
        sim_pop_activity.plot_samples(samples.T, num_neurons, folder, '', index=index)
    
    samples_r = samples.reshape((num_samples,num_neurons,num_bins))
    all_values = grad_maps[samples_r==1]
    threshold = np.percentile(all_values,threshold_prct)
    predicted_packets = np.zeros((num_samples,num_neurons*num_bins))
    for ind_s in range(samples.shape[0]):
        thr_map = grad_maps[ind_s,:,:].copy()
        thr_map[thr_map<threshold] = 0
        spikes = np.nonzero(np.sum(thr_map,axis=0))[0]
        if len(spikes)>0:
            first_spike = spikes[0]
            last_spike = spikes[-1]
            aligned_map = thr_map[:,first_spike:last_spike+1]
            aux = np.zeros((num_neurons,num_bins))
            aux[:,0:aligned_map.shape[1]] = aligned_map
            predicted_packets[ind_s,:] = aux.flatten()
    if plot_fig:
        sim_pop_activity.plot_samples(predicted_packets.T, num_neurons, folder, 'grad_maps_pred', index=index)
   
    return predicted_packets




def nearest_sample(X_real, fake_samples, num_neurons, num_bins, folder='', name='', num_samples=2**13):
    '''
    for each sample in fake_samples finds the most similar sample in X_real (if fake_samples and X_real are equal, it finds the closest sample excluding the sample itself)
    '''
    assert X_real.shape[0]==fake_samples.shape[0]
    
    if np.all(X_real.shape==fake_samples.shape):
        remove_sample = np.all(X_real==fake_samples)
    else:
        remove_sample = False
    print(remove_sample)
    fake_samples = (fake_samples > np.random.random(fake_samples.shape)).astype(float)   
    num_samples = np.min([num_samples,fake_samples.shape[1]])
    closest_sample = np.zeros((num_samples,))
    for ind_s in range(num_samples):
        if ind_s%100==0:
            print(ind_s)
        sample = fake_samples[:,ind_s]
        differences = np.sum(np.abs(X_real-sample.reshape(fake_samples.shape[0],1)),axis=0)
        if remove_sample:
            differences[ind_s] = np.inf            
        closest_sample[ind_s] = np.argmin(differences)
        
    data = {'closest_sample':closest_sample, 'samples':fake_samples}
    np.savez(folder + '/closest_sample_'+name+'.npz', **data)   
    figures.nearest_sample(num_neurons, num_bins, folder, name)   
    
        
    