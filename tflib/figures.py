#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:04:08 2017

@author: manuel
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:13:56 2017

@author: manuel
"""

import numpy as np
from tflib import  retinal_data, analysis
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots
font_size = 14
margin = 0.02
colors = ['k','g','r','b']    
    
def figure_2_3(num_samples, num_neurons, num_bins, folder, folder_fc, fig_2_or_3, neg_corrs=False, name='', main_folder=''):
    '''
    produces figures 2 (Spike-GAN VS ML), 3 (fitting retinal data) and S6 (fitting simulated data presenting negative correlations)
    '''
    original_data = np.load(folder + '/stats_real.npz')   
    mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
    [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
    
    #load conv information
    conv_data = np.load(folder + '/samples_fake.npz')['samples']
    conv_data_bin = (conv_data > np.random.random(conv_data.shape)).astype(float)   
    cov_mat_conv, k_probs_conv, mean_spike_count_conv, autocorrelogram_mat_conv, firing_average_time_course_conv, lag_cov_mat_conv = \
        analysis.get_stats_aux(conv_data_bin, num_neurons, num_bins)
    #load fc information
    if fig_2_or_3==2:
        fc_data = np.load(folder_fc + '/samples_fake.npz')['samples']
        fc_data_bin = (fc_data > np.random.random(fc_data.shape)).astype(float)   
        cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
            analysis.get_stats_aux(fc_data_bin, num_neurons, num_bins)
    elif fig_2_or_3==3:
        if neg_corrs:
            k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1', folder=main_folder+'/data/negative corrs data/')    
            cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
                analysis.get_stats_aux(k_pairwise_samples, num_neurons, num_bins)
            DDG_samples = retinal_data.load_samples_from_DDG_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1', folder=main_folder+'/data/negative corrs data/')  
            cov_mat_comp_DDG, k_probs_comp_DDG, mean_spike_count_comp_DDG, autocorrelogram_mat_comp_DDG, firing_average_time_course_comp_DDG, lag_cov_mat_comp_DDG = \
                analysis.get_stats_aux(DDG_samples, num_neurons, num_bins)
    
        else:
            k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1', folder=main_folder+'/data/retinal data/')      
            cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
                analysis.get_stats_aux(k_pairwise_samples, num_neurons, num_bins)
            DDG_samples = retinal_data.load_samples_from_DDG_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1', folder=main_folder+'/data/retinal data/')      
            cov_mat_comp_DDG, k_probs_comp_DDG, mean_spike_count_comp_DDG, autocorrelogram_mat_comp_DDG, firing_average_time_course_comp_DDG, lag_cov_mat_comp_DDG = \
                analysis.get_stats_aux(DDG_samples, num_neurons, num_bins)
    
    
    only_cov_mat_conv = cov_mat_conv.copy()
    only_cov_mat_conv[np.diag_indices(num_neurons)] = np.nan
    only_cov_mat_comp = cov_mat_comp.copy()
    only_cov_mat_comp[np.diag_indices(num_neurons)] = np.nan

    if fig_2_or_3==3:
        only_cov_mat_comp_DDG = cov_mat_comp_DDG.copy()
        only_cov_mat_comp_DDG[np.diag_indices(num_neurons)] = np.nan 
    #PLOT
    
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    if fig_2_or_3==2:
        f = plt.figure(figsize=(8, 10),dpi=250)
    else:
        f = plt.figure(figsize=(10, 6),dpi=250)
    
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    
    if fig_2_or_3==2:
        plt.subplot(3,1,1)
        num_rows = 1
        num_cols = 1
        for ind_s in range(num_rows*num_cols):
            sample = conv_data[:,ind_s].reshape((num_neurons,-1))
            sample_binnarized = conv_data_bin[:,ind_s].reshape((num_neurons,-1))
            for ind_n in range(num_neurons):
                plt.plot(sample[ind_n,:]+4*ind_n,colors[0])
                spks = np.nonzero(sample_binnarized[ind_n,:])[0]
                for ind_spk in range(len(spks)):
                    plt.plot(np.ones((2,))*spks[ind_spk],4*ind_n+np.array([2.2,3.2]),colors[2])
            #sbplt.axis('off')
            plt.axis('off')
            plt.xlim(0,num_bins)
            plt.ylim(-1,65)
            ax = plt.gca()
            points = ax.get_position().get_points()
            plt.text(points[0][0]-margin,points[1][1]+margin, 'A', fontsize=font_size, transform=plt.gcf().transFigure)
            
            
            
    if fig_2_or_3==2:
        plt.subplot(3,3,8)
    else:
        plt.subplot(2,3,5)
    #plot autocorrelogram(s)
    plt.plot(index, autocorrelogram_mat_comp,colors[1])
    if fig_2_or_3==3:
        plt.plot(index, autocorrelogram_mat_comp_DDG, colors[3], linestyle='--')
    plt.plot(index, autocorrelogram_mat_conv,colors[2])
    plt.plot(index, autocorrelogram_mat_real,colors[0])
    plt.title('autocorrelogram')
    plt.xlabel('time (ms)')
    plt.ylabel('normalized number of spikes')
    ax = plt.gca()
    points = ax.get_position().get_points()
    maximo = np.max(np.array([np.max(autocorrelogram_mat_comp),np.max(autocorrelogram_mat_conv),np.max(autocorrelogram_mat_real)]))
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'F', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'E', fontsize=font_size, transform=plt.gcf().transFigure)
        plt.annotate('Ground truth',xy=(-9.5,maximo+4*maximo/10),fontsize=8,color=colors[0])
        plt.annotate('k-pairwise model',xy=(-9.5,maximo+3*maximo/10),fontsize=8,color=colors[1])
        plt.annotate('DG model',xy=(-9.5,maximo+2*maximo/10),fontsize=8,color=colors[3])  
        plt.annotate('Spike-GAN',xy=(-9.5,maximo+maximo/10),fontsize=8,color=colors[2])
        plt.ylim(0,maximo+5*maximo/10)

    #plot mean firing rates
    mean_spike_count_real = mean_spike_count_real*1000/num_bins
    mean_spike_count_conv = mean_spike_count_conv*1000/num_bins
    mean_spike_count_comp = mean_spike_count_comp*1000/num_bins
    if fig_2_or_3==3:
        mean_spike_count_comp_DDG = mean_spike_count_comp_DDG*1000/num_bins
    if fig_2_or_3==2:
        plt.subplot(3,3,4)
    else:
        plt.subplot(2,3,1)
        if not neg_corrs:
            mean_spike_count_real = mean_spike_count_real/20
            mean_spike_count_conv = mean_spike_count_conv/20
            mean_spike_count_comp = mean_spike_count_comp/20
            mean_spike_count_comp_DDG = mean_spike_count_comp_DDG/20
    maximo = np.max(np.array([np.max(mean_spike_count_real),np.max(mean_spike_count_conv),np.max(mean_spike_count_comp)]))
    minimo = np.min(np.array([np.min(mean_spike_count_real),np.min(mean_spike_count_conv),np.min(mean_spike_count_comp)]))
    maximo += maximo/20
    minimo -= maximo/20
    axis_ticks = np.linspace(minimo,maximo,3)
    plt.plot([minimo,maximo],[minimo,maximo],colors[0])
    plt.plot(mean_spike_count_real,mean_spike_count_conv,'.'+colors[2])
    plt.plot(mean_spike_count_real,mean_spike_count_comp,'x'+colors[1])
    if fig_2_or_3==3:
        plt.plot(mean_spike_count_real,mean_spike_count_comp_DDG,'+'+colors[3])
    plt.xlabel('mean firing rate expt (Hz)')
    plt.ylabel('mean firing rate models (Hz)')   
    plt.title('mean firing rates')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'B', fontsize=font_size, transform=plt.gcf().transFigure)
        plt.annotate('Spike-GAN',xy=(minimo,maximo-(maximo-minimo)/10),fontsize=8,color=colors[2])
        plt.annotate('MLP GAN',xy=(minimo,maximo-2*(maximo-minimo)/10),fontsize=8,color=colors[1])
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'A', fontsize=font_size, transform=plt.gcf().transFigure)
        plt.annotate('k-pairwise model',xy=(minimo,maximo-2*(maximo-minimo)/10),fontsize=8,color=colors[1])
        plt.annotate('DG model',xy=(minimo,maximo-3*(maximo-minimo)/10),fontsize=8,color=colors[3])
        plt.annotate('Spike-GAN',xy=(minimo,maximo-(maximo-minimo)/10),fontsize=8,color=colors[2])
        
        

    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #plot covariances
    if fig_2_or_3==2:
        plt.subplot(3,3,5)
    else:
        plt.subplot(2,3,2)
    only_cov_mat_real = cov_mat_real.copy()
    only_cov_mat_real[np.diag_indices(num_neurons)] = np.nan
    axis_ticks = np.linspace(np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten()),3)
    plt.plot([np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],\
                    [np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],colors[0])
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_conv.flatten(),'.'+colors[2])
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_comp.flatten(),'x'+colors[1])
    if fig_2_or_3==3:
        plt.plot(only_cov_mat_real.flatten(),only_cov_mat_comp_DDG.flatten(),'+'+colors[3])
    plt.title('pairwise covariances')
    plt.xlabel('covariances expt')
    plt.ylabel('covariances models')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'C', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'B', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    #plot k-statistics
    if fig_2_or_3==2:
        plt.subplot(3,3,6)
    else:
        plt.subplot(2,3,3)
    axis_ticks = np.linspace(0,np.max(k_probs_real),3)
    plt.plot([0,np.max(k_probs_real)],[0,np.max(k_probs_real)],colors[0])        
    plt.plot(k_probs_real,k_probs_conv,'.'+colors[2])        
    plt.plot(k_probs_real,k_probs_comp,'x'+colors[1]) 
    if fig_2_or_3==3:
        plt.plot(k_probs_real,k_probs_comp_DDG,'+'+colors[3])
    plt.xlabel('k-probs expt')
    plt.ylabel('k-probs models')
    plt.title('k statistics') 
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'D', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'C', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    #plot average time course
    #firing_average_time_course[firing_average_time_course>0.048] = 0.048
    title_size = 8
    firing_average_time_course_real_section = firing_average_time_course_real*1000
    firing_average_time_course_conv_section = firing_average_time_course_conv*1000
    firing_average_time_course_comp_section = firing_average_time_course_comp*1000
    if fig_2_or_3==3:
        firing_average_time_course_comp_section_DDG = firing_average_time_course_comp_DDG*1000
    if fig_2_or_3==2:
        plt.subplot(9,5,31)
    else:
        plt.subplot(8,3,13)
        if not neg_corrs:
            firing_average_time_course_real_section = firing_average_time_course_real_section/20
            firing_average_time_course_conv_section = firing_average_time_course_conv_section/20
            firing_average_time_course_comp_section = firing_average_time_course_comp_section/20
            firing_average_time_course_comp_section_DDG = firing_average_time_course_comp_section_DDG/20
        
    
    maximo = np.max(firing_average_time_course_real_section.flatten())
    minimo = np.min(firing_average_time_course_real_section.flatten())
    if fig_2_or_3==2:
        aspect_ratio = firing_average_time_course_real_section.shape[1]/(1.7*firing_average_time_course_real_section.shape[0])
    else:
        aspect_ratio = firing_average_time_course_real_section.shape[1]/(2*firing_average_time_course_real_section.shape[0])
    plt.imshow(firing_average_time_course_real_section,interpolation='nearest', cmap='viridis', aspect=aspect_ratio)
    plt.title('real data', fontsize=title_size)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'E', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'D', fontsize=font_size, transform=plt.gcf().transFigure)
    if fig_2_or_3==2:
        plt.subplot(9,5,36)
    else:
        plt.subplot(8,3,16)
    
    plt.imshow(firing_average_time_course_conv_section,interpolation='nearest', clim=(minimo,maximo), cmap='viridis',aspect=aspect_ratio)
    plt.title('Spike-GAN', fontsize=title_size)
    plt.xticks([])
    plt.yticks([])
    if fig_2_or_3==2:
        plt.subplot(9,5,41)
    else:
        plt.subplot(8,3,19)
    
    map_aux = plt.imshow(firing_average_time_course_comp_section,interpolation='nearest', clim=(minimo,maximo), cmap='viridis',aspect=aspect_ratio)#map_aux = 
    if fig_2_or_3==2:
        plt.title('MLP GAN', fontsize=title_size)
    else:
        plt.title('k-pairwise', fontsize=title_size)
    if fig_2_or_3==2:
        plt.xlabel('time')
        plt.ylabel('neuron')
        ax = plt.gca()
    plt.xticks([])
    plt.yticks([])
        
    
    
    if fig_2_or_3==3:
        plt.subplot(8,3,22)    
        map_aux = plt.imshow(firing_average_time_course_comp_section_DDG,interpolation='nearest', clim=(minimo,maximo), cmap='viridis',aspect=aspect_ratio)#map_aux = 
        plt.title('DG', fontsize=title_size)
        plt.xlabel('time')
        plt.ylabel('neuron')
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
    
    points_colorbar = ax.get_position().get_points()
    size_colorbar = ax.get_position().size
    ticks_values = [np.floor(minimo)+1, np.floor((maximo+minimo)/2), np.floor(maximo)]
    if fig_2_or_3==2:
        #cbaxes = f.add_axes([points_colorbar[0][0]+(points_colorbar[1][0]-points_colorbar[0][0])/6, points_colorbar[0][1]-0.03, (points_colorbar[1][0]-points_colorbar[0][0])/1.5, 0.01]) 
        cbaxes = f.add_axes([points_colorbar[0][0]+1.2*size_colorbar[0], 1.05*points_colorbar[0][1], 0.01, 0.1]) 
        plt.colorbar(map_aux, orientation='vertical', cax = cbaxes, ticks=ticks_values)    
    else:
        #cbaxes = f.add_axes([points_colorbar[0][0]+(points_colorbar[1][0]-points_colorbar[0][0])/6, points_colorbar[0][1]-0.05, (points_colorbar[1][0]-points_colorbar[0][0])/1.5, 0.01]) 
        cbaxes = f.add_axes([points_colorbar[0][0]+size_colorbar[0]/1.2, points_colorbar[0][1], 0.01, 0.15]) 
        plt.colorbar(map_aux, orientation='vertical', cax = cbaxes, ticks=ticks_values)    
    cbaxes.set_ylabel('firing rate (Hz)',rotation=-90)
    #cbaxes.yaxis.set_label_coords(-0.3,0.5)
    #plot lag covariance
    if fig_2_or_3==2:
        plt.subplot(3,3,9)
    else:
        plt.subplot(2,3,6)
    axis_ticks = np.linspace(np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten()),3)
    plt.plot([np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],\
                        [np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],colors[0])        
    plt.plot(lag_cov_mat_real,lag_cov_mat_conv,'.'+colors[2])
    plt.plot(lag_cov_mat_real,lag_cov_mat_comp,'x'+colors[1])
    if fig_2_or_3==3:
        plt.plot(lag_cov_mat_real,lag_cov_mat_comp_DDG,'+'+colors[3])
    plt.xlabel('lag cov real')
    plt.ylabel('lag cov models')
    plt.title('lag covariances')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_3==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'G', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'F', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    
    f.savefig(folder+name+'_II.svg',dpi=600, bbox_inches='tight')
    f.savefig(main_folder+'/figures paper/'+name+'.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    return points_colorbar, cbaxes, map_aux, maximo, minimo

def figure_4(num_samples, num_neurons, num_bins, folder, num_rows=1, sample_dir = '', main_folder=''):
    '''
    produces figure 4 (importance maps) and S8 (more importance maps)
    '''
    colors = np.divide(np.array([(0, 0, 0), (128, 128, 128),(166,206,227),(31,120,180),(51,160,44),(251,154,153),(178,223,138)]),256)
    cm = LinearSegmentedColormap.from_list('my_map', colors, N=7)
    
    #original_data = np.load(folder + '/stats_real.npz') 
    importance_maps = np.load(folder+'importance_vectors_1_8_8000.npz')
    f = plt.figure(figsize=(8, 10),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    num_cols = 5
    
    if num_rows==1 and folder.find('retina')==-1:
        packet = np.load(folder+'packet.npz')
        maximo = np.max(packet['packet'].flatten())
        cbaxes = f.add_axes([0.1,0.85,0.4,0.25]) 
        plt.imshow(packet['packet'],interpolation='nearest',clim=[0,maximo],cmap=cm)
        plt.xlabel('time (ms)')
        plt.ylabel('neurons')
        plt.title('ideal packet')
        ax = plt.gca()
        points_colorbar = ax.get_position().get_points()
        plt.text(0.04,1.075, 'A', fontsize=14, transform=plt.gcf().transFigure)
        cbaxes = f.add_axes([0.57,0.85,0.4,0.25]) 
        plt.imshow(packet['result'],interpolation='nearest',clim=[0,maximo],cmap=cm)
        plt.axis('off')
        plt.title('realistic population activity')
        plt.text(0.55,1.075, 'B', fontsize=14, transform=plt.gcf().transFigure)
        #title
        plt.text(0.53,points_colorbar[0][1]-0.005, 'Importance Maps', ha='center', fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.04,points_colorbar[0][1]-0.005, 'C', fontsize=14, transform=plt.gcf().transFigure)
    else:
        points_colorbar = np.array([[0.01,0.095],0,0])
    #index = np.argsort(original_data['shuffled_index'])
   
    num_samples = num_cols*num_rows
    if num_rows==1:
        grad_maps = importance_maps['grad_maps'][[0,1,5,6,7],:,:]
        samples = importance_maps['samples'][[0,1,5,6,7],:]
    else:
        aux = np.arange(importance_maps['grad_maps'].shape[0])#range(num_samples)
        np.random.shuffle(aux)
        aux = aux[0:num_samples]
        grad_maps = importance_maps['grad_maps'][aux,:,:]
        samples = importance_maps['samples'][aux,:]
        
    #panels position params
    if folder.find('retina')!=-1:
        width = 0.4
        height = width*num_bins/(3*num_neurons)
        margin = width/10
        factor_width = 0.25
        factor_height = 0
    else:
        width = 0.17
        height = width*num_bins/(3*num_neurons)
        margin = width/10
        factor_width = 0
        factor_height = 0.045
        
    for i in range(num_samples):
        pos_h = (points_colorbar[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)
        cbaxes = f.add_axes([pos_h, pos_v, width, height]) 
        sample = samples[i,:]
        sample = sample.reshape(num_neurons,num_bins)
        cbaxes.imshow(sample,interpolation='nearest',clim=[0,np.max(samples.flatten())],cmap='gray')
        cbaxes.axis('off')  
        
        pos_h = (points_colorbar[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)-height+factor_height
        cbaxes = f.add_axes([pos_h,pos_v, width, height]) 
        cbaxes.imshow(grad_maps[i,:,:],interpolation='nearest', cmap = plt.cm.hot, clim=[0,np.max(grad_maps.flatten())])  
        cbaxes.axis('off')  
        reference = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)-height+factor_height
    
    if num_rows==1:
        importance_time_vector = importance_maps['time']
        importance_neuron_vector = importance_maps['neurons']
        importance_time_vector_surr = importance_maps['time_surr']
        importance_neuron_vector_surr = importance_maps['neurons_surr']
        cbaxes = f.add_axes([0.1,reference-0.3,0.4,0.25]) 
        plt.errorbar(np.arange(num_bins), np.mean(importance_time_vector,axis=0), yerr=np.std(importance_time_vector,axis=0)/np.sqrt(importance_time_vector.shape[0]))
        plt.errorbar(np.arange(num_bins), np.mean(importance_time_vector_surr,axis=0), yerr=np.std(importance_time_vector_surr,axis=0)/np.sqrt(importance_time_vector_surr.shape[0]),color=(.7,.7,.7))
        plt.ylabel('average importance (a.u.)')
        plt.xlabel('time (ms)')
        plt.title('importance of different time periods')
        plt.xlim(-1,num_bins)
        plt.text(0.04,reference-0.025, 'D', fontsize=14, transform=plt.gcf().transFigure)
        cbaxes = f.add_axes([0.57,reference-0.3,0.4,0.25]) 
        plt.bar(np.arange(num_neurons), np.mean(importance_neuron_vector,axis=0), yerr=np.std(importance_neuron_vector,axis=0)/np.sqrt(importance_neuron_vector.shape[0]))
        plt.errorbar(np.arange(num_neurons)+0.5, np.mean(importance_neuron_vector_surr,axis=0), yerr=np.std(importance_neuron_vector_surr,axis=0)/np.sqrt(importance_neuron_vector_surr.shape[0]),color=(1,0,0))#,fill=False,edgecolor=(1,0,0))
        plt.xlabel('neurons')
        plt.title('importance of different neurons')
        plt.xlim(-1,num_neurons+1)
        plt.text(0.55,reference-0.025, 'E', fontsize=14, transform=plt.gcf().transFigure)
        f.savefig(sample_dir+'figure_4_reduced_1_8_8000.svg',dpi=600, bbox_inches='tight')
        f.savefig(main_folder + '/figures paper/figure_4_1_8_8000.svg',dpi=600, bbox_inches='tight')
    else:
        f.savefig(sample_dir+'figure_4_many_samples_1_8_8000.svg',dpi=600, bbox_inches='tight')
        f.savefig(main_folder+'/figures paper/figure_4_many_samples_1_8_8000.svg',dpi=600, bbox_inches='tight')
    plt.close(f)

    activity_map = importance_maps['activity_map']
    f = plt.figure(figsize=(8, 10),dpi=250)
    cbaxes = f.add_axes([0.1,0.1,0.4,0.7]) 
    plt.errorbar(np.arange(num_bins), np.mean(activity_map,axis=0), yerr=np.std(activity_map,axis=0)/np.sqrt(activity_map.shape[0]))
    plt.ylabel('average importance (a.u.)')
    plt.xlabel('time (ms)')
    plt.title('activity of different time periods')
    plt.xlim(-1,num_bins)
    plt.text(0.04,reference-0.025, 'D', fontsize=14, transform=plt.gcf().transFigure)
    cbaxes = f.add_axes([0.57,0.1,0.4,0.7]) 
    plt.bar(np.arange(num_neurons), np.mean(activity_map,axis=1), yerr=np.std(activity_map,axis=1)/np.sqrt(activity_map.shape[1]))
    plt.xlabel('neurons')
    plt.title('activity of different neurons')
    plt.xlim(-1,num_neurons+1)
    plt.text(0.55,reference-0.025, 'E', fontsize=14, transform=plt.gcf().transFigure)
    f.savefig(sample_dir+'average_activity_1_8_8000.svg',dpi=600, bbox_inches='tight')

def figure_supp_using_imp_maps(folder, num_samples, name='', noise=0, main_folder=''):
    '''
    produces fig S2 and S9 (using reliably/noisy importance maps)
    '''
    noisy_packet = noise>0
    real_data = np.load(folder + '/stats_real.npz')
    stimulus_id = np.load(folder + '/stim.npz')['stimulus']
    importance_info = np.load(folder + '/importance_vectors_1_4_'+str(num_samples)+'.npz')
    grads = importance_info['grad_maps']
    num_samples = grads.shape[0]
    num_neurons = grads.shape[1]
    num_bins = grads.shape[2]
    samples = importance_info['samples']
    index = np.argsort(real_data['shuffled_index'])#np.arange(num_neurons)#
    ground_truth_packets = np.load(folder + '/packets.npz')
    ground_truth_packets_stim1 = analysis.find_packets(ground_truth_packets['packets_stim1'], ground_truth_packets['packets_stim1']-1, num_neurons, num_bins, folder, int(num_samples/2), plot_fig=False)
    ground_truth_packets_stim2 = analysis.find_packets(ground_truth_packets['packets_stim2'], ground_truth_packets['packets_stim2']-1, num_neurons, num_bins, folder, int(num_samples/2), plot_fig=False)
    ground_truth_packets_stim1 = np.mean(ground_truth_packets_stim1, axis=0)
    ground_truth_packets_stim2 = np.mean(ground_truth_packets_stim2, axis=0)
    
    stimulus_id = stimulus_id[0:num_samples]
    
    predicted_packets = analysis.find_packets(grads,samples,num_neurons, num_bins, folder, num_samples, threshold_prct=50)
    
    f = plt.figure(figsize=(10, 6),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    #PLOT
    if noisy_packet:
        plot_noisy_packets(samples, stimulus_id, index, f, ground_truth_packets_stim1, ground_truth_packets_stim2, num_neurons, num_bins)    
    else:
        cbaxes = plt.subplot(2,2,1)
        plt.text(0.1,0.9, 'A', fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.19,0.88, 'Hypothetical Experiment', fontsize=12, transform=plt.gcf().transFigure)
        plt.axis('off')
        #samples
        cbaxes1 = plt.subplot(4,6,4)
        cbaxes2 = plt.subplot(4,6,10)
        letter = 'B'
        title = 'sorted samples'
        cmap = 'gray'
        plot_samples_aux(samples, stimulus_id, index, f, cbaxes1, cbaxes2, letter, title, cmap, num_neurons, num_bins)
            
    
        #shuffled samples
        cbaxes1 = plt.subplot(4,6,5)
        cbaxes2 = plt.subplot(4,6,11)
        letter = 'C'
        title = 'recorded samples'
        cmap = 'gray'
        plot_samples_aux(samples, stimulus_id, np.arange(num_neurons), f, cbaxes1, cbaxes2, letter, title, cmap, num_neurons, num_bins)
    
        
        #importance maps 
        cbaxes1 = plt.subplot(4,6,6)
        cbaxes2 = plt.subplot(4,6,12)
        letter = 'D'
        title = 'importance maps'
        cmap = plt.cm.hot
        plot_samples_aux(grads, stimulus_id, np.arange(num_neurons), f, cbaxes1, cbaxes2, letter, title, cmap, num_neurons, num_bins)
    
    
    #average packets 
    packet1_ind, packet2_ind = plot_inferred_packets(real_data, predicted_packets, stimulus_id, ground_truth_packets_stim1, ground_truth_packets_stim2, num_neurons, num_bins, folder, f, noisy_packet)
    
    #importance neurons
    #importance_neuron_vector_surr = importance_info['neurons_surr']
    cbaxes = f.add_axes([0.45, 0.125, 0.45, 0.3])
    bar_width = 0.35
    stim_1_grads = grads[stimulus_id==1,:,:]
    mean_stim1 = np.mean(np.mean(stim_1_grads,axis=2),axis=0)
    cbaxes.bar(np.arange(num_neurons), mean_stim1, bar_width, color=colors[2], edgecolor='w', label='Stim1')
    stim_2_grads = grads[stimulus_id==2,:,:]
    mean_stim2 = np.mean(np.mean(stim_2_grads,axis=2),axis=0)
    maximo = np.max(np.array([np.max(mean_stim1),np.max(mean_stim2)]))+np.max(np.array([np.max(mean_stim1),np.max(mean_stim2)]))/30
    cbaxes.bar(np.arange(num_neurons)+ bar_width, mean_stim2, bar_width, color='b', edgecolor='w', label='Stim2')
    plt.plot(packet1_ind[0]+bar_width, np.zeros((packet1_ind[0].shape[0],))+maximo,'+r')
    plt.plot(packet2_ind[0]+bar_width, np.zeros((packet2_ind[0].shape[0],))+maximo,'xb')
    
    plot_threshold = 1
    if plot_threshold:
        #significance_th = 99
        threshold_stim1 = np.ones((num_neurons,))*0.0065#np.percentile(importance_neuron_vector_surr[stimulus_id==1,:],significance_th,axis=0)
        plt.plot(np.arange(num_neurons)+bar_width/2, threshold_stim1, linestyle='--', color=(0.75,0.75,0.75))
        threshold_stim2 = threshold_stim1#np.percentile(importance_neuron_vector_surr[stimulus_id==2,:],significance_th,axis=0)
        #plt.plot(np.arange(num_neurons)+3*bar_width/2, threshold_stim2, marker='', color=(0,0,0.75))
        aux = mean_stim1>threshold_stim1
        recall = np.sum(aux[packet1_ind[0]])/len(packet1_ind[0])
        plt.text(0.475,0.4, 'recall: '+ "{0:.2f}".format(round(recall,2)), fontsize=8, color=(1,0,0), transform=plt.gcf().transFigure)
        precision = np.sum(aux[packet1_ind[0]])/np.sum(aux)
        plt.text(0.585,0.4, 'precision: '+ "{0:.2f}".format(round(precision,2)), fontsize=8, color=(1,0,0), transform=plt.gcf().transFigure)
        
      
        aux = mean_stim2>threshold_stim2
        recall = np.sum(aux[packet2_ind[0]])/len(packet2_ind[0])
        plt.text(0.475,0.37, 'recall: '+ "{0:.2f}".format(round(recall,2)), fontsize=8, color=(0,0,1), transform=plt.gcf().transFigure)
        precision = np.sum(aux[packet2_ind[0]])/np.sum(aux)
        plt.text(0.585,0.37, 'precision: '+ "{0:.2f}".format(round(precision,2)), fontsize=8, color=(0,0,1), transform=plt.gcf().transFigure)
    
            
    cbaxes.set_ylabel('average importance (a.u.)')
    cbaxes.set_xlabel('neuron ID')
    cbaxes.set_title('importance of different neurons')
    cbaxes.set_xlim(-1,num_neurons+1)
    maximo = np.max(np.array([np.max(mean_stim1),np.max(mean_stim2)]))+np.max(np.array([np.max(mean_stim1),np.max(mean_stim2)]))/1.5
    cbaxes.set_ylim(0,maximo)
    cbaxes.legend()
    if noisy_packet:
        plt.text(0.43,0.45, 'C', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(0.43,0.45, 'F', fontsize=font_size, transform=plt.gcf().transFigure)
    f.savefig(folder+'supp_fig_using_importance_maps.svg',dpi=600, bbox_inches='tight')
    f.savefig(main_folder + '/figures paper/supp_fig_using_importance_maps'+str(num_samples)+name+'.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    
    
def plot_inferred_packets(real_data, predicted_packets, stimulus_id, ground_truth_packets_1, ground_truth_packets_2, num_neurons, num_bins, folder, f, noisy_packet):
    '''
    produces figs S2E and S9B
    '''
    ground_truth_packets_1 = ground_truth_packets_1.reshape((num_neurons,num_bins))
    packet1_or = ground_truth_packets_1[real_data['shuffled_index'],0:21]
    
    packet1 = np.mean(predicted_packets[stimulus_id==1,:],axis=0)
    packet1 = packet1.reshape((num_neurons,num_bins))
    packet1 /= np.max(packet1)
    packet1 = packet1[:,0:21]
    correlation_1 = np.corrcoef(packet1_or.flatten(),packet1.flatten()).T
    
    ground_truth_packets_2 = ground_truth_packets_2.reshape((num_neurons,num_bins))
    packet2_or = ground_truth_packets_2[real_data['shuffled_index'],0:21]
    
    packet2 = np.mean(predicted_packets[stimulus_id==2,:],axis=0)
    packet2 = packet2.reshape((num_neurons,num_bins))
    packet2 /= np.max(packet2)
    packet2 = packet2[:,0:21]
    correlation_2 = np.corrcoef(packet2_or.flatten(),packet2.flatten()).T


    
    packet = np.load(folder + '/packet0.npz')
    packet = packet['sample']
    packet[packet==1] = 0
    packet[packet==2] = 1   
    spikes = np.nonzero(np.sum(packet,axis=0))[0]
    first_spike = spikes[0] 
    last_spike = spikes[-1]
    aligned_map = packet[:,first_spike:last_spike+1]
    packet1_or = np.zeros((num_neurons,num_bins))
    packet1_or[:,0:aligned_map.shape[1]] = aligned_map
    packet1_or = packet1_or[real_data['shuffled_index'],0:21]
    packet1_ind = np.nonzero(packet1_or==1)
    
    packet = np.load(folder + '/packet1.npz')
    packet = packet['sample']
    packet[packet==1] = 0
    packet[packet==2] = 1
    packet[packet==2] = 1   
    spikes = np.nonzero(np.sum(packet,axis=0))[0]
    first_spike = spikes[0] 
    last_spike = spikes[-1]
    aligned_map = packet[:,first_spike:last_spike+1]
    packet2_or = np.zeros((num_neurons,num_bins))
    packet2_or[:,0:aligned_map.shape[1]] = aligned_map
    packet2_or = packet2_or[real_data['shuffled_index'],0:21]
    packet2_ind = np.nonzero(packet2_or==1)
    

    position1 = [0.1,0.12]
    cbaxes = f.add_axes([position1[0], position1[1], 0.2, 0.3])
    cbaxes.imshow(packet1,interpolation='nearest', cmap='gray') 
    cbaxes.plot(packet1_ind[1],packet1_ind[0],'+g')
    if noisy_packet:
        plt.text(position1[0],0.45, 'B', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(position1[0],0.45, 'E', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.text(position1[0]+0.1,0.45, 'template packets', fontsize=12, transform=plt.gcf().transFigure)
    if noisy_packet:
        plt.text(position1[0]+0.055,position1[1]+0.25, "{0:.2f}".format(round(correlation_1[1,0],2)), fontsize=8, transform=plt.gcf().transFigure, color='y')
    else:
        plt.text(position1[0]+0.105,position1[1]+0.25, "r={0:.2f}".format(round(correlation_1[1,0],2)), fontsize=8, transform=plt.gcf().transFigure, color='y')
    plt.text(position1[0]+0.08,position1[1]+0.28, 'stim1', fontsize=12, transform=plt.gcf().transFigure, color='r')
    plt.axis('off')
    
    position2 = [0.22,0.12]
    cbaxes = f.add_axes([position2[0], position2[1], 0.2, 0.3])
    cbaxes.imshow(packet2,interpolation='nearest', cmap='gray')  
    cbaxes.plot(packet2_ind[1],packet2_ind[0],'+g')
    if noisy_packet:
        plt.text(position2[0]+0.121,position2[1]+0.25, "{0:.2f}".format(round(correlation_2[1,0],2)), fontsize=8, transform=plt.gcf().transFigure, color='y')
    else:
        plt.text(position2[0]+0.105,position2[1]+0.25, "r={0:.2f}".format(round(correlation_2[1,0],2)), fontsize=8, transform=plt.gcf().transFigure, color='y')
    plt.text(position2[0]+0.08,position2[1]+0.28, 'stim2', fontsize=12, transform=plt.gcf().transFigure, color='b')
    plt.axis('off')   
    return packet1_ind, packet2_ind
    

def plot_samples_aux(samples, stimulus_id, index, f, cbaxes1, cbaxes2, letter, title, cmap, num_neurons, num_bins):
    '''
    figure S2B-D
    '''
    shift = 0.02
    width = 0.2
    line_color = 'yellow'
    margin = 0.03
    font_titles = 8
    ind_dc = 2
    samples_1 = samples[stimulus_id==1,:]
    sample = samples_1[0+ind_dc,:]
    sample = sample.reshape(num_neurons,num_bins)
    sample = sample[index,:]
    cbaxes1.imshow(sample,interpolation='nearest', cmap=cmap)   
    box_lines(cbaxes1, line_color, width)
    points = cbaxes1.get_position().get_points()
    size = cbaxes1.get_position().size
    cbaxes = f.add_axes([points[0,0]-shift, points[0,1]-shift, size[0], size[1]])
    sample = samples_1[1+ind_dc,:]
    sample = sample.reshape(num_neurons,num_bins)
    sample = sample[index,:]
    cbaxes.imshow(sample,interpolation='nearest', cmap=cmap) 
    box_lines(cbaxes, line_color, width)
    plt.text(points[0][0]-margin,points[1][1], letter, fontsize=font_size, transform=plt.gcf().transFigure)
    plt.text(points[0][0],points[1][1]+margin/2, title, fontsize=font_titles, transform=plt.gcf().transFigure)
    
    samples_2 = samples[stimulus_id==2,:]
    sample = samples_2[0+ind_dc,:]
    sample = sample.reshape(num_neurons,num_bins)
    sample = sample[index,:]
    cbaxes2.imshow(sample,interpolation='nearest', cmap=cmap)   
    box_lines(cbaxes2, line_color, width)
    points = cbaxes2.get_position().get_points()
    size = cbaxes2.get_position().size
    cbaxes = f.add_axes([points[0,0]-shift, points[0,1]-shift, size[0], size[1]])
    sample = samples_2[1+ind_dc,:]
    sample = sample.reshape(num_neurons,num_bins)
    sample = sample[index,:]
    cbaxes.imshow(sample,interpolation='nearest', cmap=cmap) 
    box_lines(cbaxes, line_color, width)
    
def plot_noisy_packets(samples, stimulus_id, index, f, ground_truth_packets1, ground_truth_packets2, num_neurons, num_bins):
    '''
    figure S9A
    '''
    plt.text(0.1,0.8, 'A', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.axis('off')
    width = 0.15
    height = 0.15
    margin =0.02
    pos_h = 0.125
    pos_v = 0.65
    num_cols = 8
    line_widt = 0.5
    samples_1 = samples[stimulus_id==1,:]
    samples_2 = samples[stimulus_id==2,:]
    for ind in range(num_cols):
        if ind<num_cols/2:
            indice = ind
            samples_temp = samples_1
            line_color = 'r'
        else:
            indice = int(ind-num_cols/2)
            samples_temp = samples_2
            line_color = 'b'
        sample = samples_temp[2*indice+1,:]
        sample = sample.reshape(num_neurons,num_bins)
        sample = sample[index,:]
        cbaxes = f.add_axes([pos_h, pos_v, width, height]) 
        cbaxes.imshow(sample,interpolation='nearest', cmap='gray') 
        box_lines(cbaxes, line_color, line_widt)
        
        sample = samples_temp[2*indice,:]
        sample = sample.reshape(num_neurons,num_bins)
        sample = sample[index,:]
        pos_v_temp = pos_v-height-margin/4
        cbaxes = f.add_axes([pos_h, pos_v_temp, width, height]) 
        cbaxes.imshow(sample,interpolation='nearest', cmap='gray') 
        box_lines(cbaxes, line_color, line_widt)
       
        pos_h += width/1.6
   

def box_lines(cbaxes, line_color, width):
    '''
    modifies panels in Fig. 2SB-D
    '''
    cbaxes.spines['top'].set_color(line_color)
    cbaxes.spines['bottom'].set_color(line_color)
    cbaxes.spines['right'].set_color(line_color)
    cbaxes.spines['left'].set_color(line_color)
    cbaxes.spines['top'].set_linewidth(width)
    cbaxes.spines['bottom'].set_linewidth(width)
    cbaxes.spines['right'].set_linewidth(width)
    cbaxes.spines['left'].set_linewidth(width)
    cbaxes.set_xticks([])
    cbaxes.set_yticks([])


def figure_supp_using_imp_maps_ROC_curves(main_folder=''):
    '''
    figure S10
    '''
    num_samples_mat = [4096, 2048, 1024]
    f = plt.figure(figsize=(10, 6),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        
    lw = 2
    letters = 'ABC'
    for ind in range(len(num_samples_mat)):
        num_s = num_samples_mat[ind]
        folder = main_folder + '/samples conv/dataset_packets_num_samples_'+str(num_s)+'_num_neurons_32_num_bins_32_packet_prob_1.0_firing_rate_0.05_group_size_18_noise_in_packet_0.0_number_of_modes_2_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_20/'    
        real_data = np.load(folder + '/stats_real.npz')
        stimulus_id = np.load(folder + '/stim.npz')['stimulus']
        importance_info = np.load(folder + '/importance_vectors_1_4_'+str(num_s)+'.npz')
        grads = importance_info['grad_maps']
        num_samples = grads.shape[0]
        num_neurons = grads.shape[1]
        num_bins = grads.shape[2]
        samples = importance_info['samples']
        ground_truth_packets = np.load(folder + '/packets.npz')
        ground_truth_packets_stim1 = analysis.find_packets(ground_truth_packets['packets_stim1'], ground_truth_packets['packets_stim1']-1, num_neurons, num_bins, folder, int(num_s/2), plot_fig=False)
        ground_truth_packets_stim2 = analysis.find_packets(ground_truth_packets['packets_stim2'], ground_truth_packets['packets_stim2']-1, num_neurons, num_bins, folder, int(num_s/2), plot_fig=False)
        ground_truth_packets_stim1 = np.mean(ground_truth_packets_stim1, axis=0)
        ground_truth_packets_stim2 = np.mean(ground_truth_packets_stim2, axis=0)
        
        stimulus_id = stimulus_id[0:num_samples]
        
        predicted_packets = analysis.find_packets(grads,samples,num_neurons, num_bins, folder, num_samples, threshold_prct=50)
        
        
        #average packets 
        fpr, tpr, roc_auc, all_stim, y_true = ROC_curve(real_data, predicted_packets, stimulus_id,  num_neurons, num_bins, folder)
        plt.subplot(1,2,2)
        aux = plt.plot(fpr, tpr, lw=lw, label='Num. Samples: %i (AUC = %0.2f)' % (num_s,roc_auc))
        cbaxes = plt.subplot(3,2,2*ind+1)
        hist_0, bins_0 = np.histogram(all_stim[y_true==0],bins=np.linspace(0,1,num=10))
        hist_0 = hist_0/np.sum(hist_0)
        hist_1, bins_1 = np.histogram(all_stim[y_true==1],bins=np.linspace(0,1,num=10))
        hist_1 = hist_1/np.sum(hist_1)
        plt.plot(bins_0[0:-1]+(bins_0[1]-bins_0[0])/2,hist_0, color=(.7,.7,.7),label='non participating bins')
        plt.plot(bins_1[0:-1]+(bins_1[1]-bins_1[0])/2,hist_1, color=(0,.0,0),label='participating bins')
        plt.title('Num. Samples: %i ' % num_s, color=aux[0].get_color())
        if ind==2:
            plt.xlabel('Normalized Importance values')
            plt.ylabel('Frequency')
        
        points = cbaxes.get_position().get_points()
        plt.text(points[0][0]-0.05,points[1][1]+0.025, letters[ind], fontsize=font_size, transform=plt.gcf().transFigure)
                
       
    cbaxes = plt.subplot(1,2,2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    points = cbaxes.get_position().get_points()
    plt.text(points[0][0]-0.05,points[1][1]+0.025, 'D', fontsize=font_size, transform=plt.gcf().transFigure)
        
    plt.subplot(3,2,1)
    plt.legend(loc="upper right")
    f.savefig(main_folder + '/figures paper/ROC_curves.svg',dpi=600, bbox_inches='tight')
    plt.close(f)


def ROC_curve(real_data, predicted_packets, stimulus_id,  num_neurons, num_bins, folder):
    '''
    computes the ROC curves for Fig. S10
    '''
    packet1 = np.mean(predicted_packets[stimulus_id==1,:],axis=0)
    packet1 = packet1.reshape((num_neurons,num_bins))
    packet1 /= np.max(packet1)
    packet1 = packet1[:,0:21]
    
    
    packet2 = np.mean(predicted_packets[stimulus_id==2,:],axis=0)
    packet2 = packet2.reshape((num_neurons,num_bins))
    packet2 /= np.max(packet2)
    packet2 = packet2[:,0:21]


    
    packet = np.load(folder + '/packet0.npz')
    packet = packet['sample']
    packet[packet==1] = 0
    packet[packet==2] = 1   
    spikes = np.nonzero(np.sum(packet,axis=0))[0]
    first_spike = spikes[0] 
    last_spike = spikes[-1]
    aligned_map = packet[:,first_spike:last_spike+1]
    packet1_or = np.zeros((num_neurons,num_bins))
    packet1_or[:,0:aligned_map.shape[1]] = aligned_map
    packet1_or = packet1_or[real_data['shuffled_index'],0:21]
    
    packet = np.load(folder + '/packet1.npz')
    packet = packet['sample']
    packet[packet==1] = 0
    packet[packet==2] = 1
    packet[packet==2] = 1   
    spikes = np.nonzero(np.sum(packet,axis=0))[0]
    first_spike = spikes[0] 
    last_spike = spikes[-1]
    aligned_map = packet[:,first_spike:last_spike+1]
    packet2_or = np.zeros((num_neurons,num_bins))
    packet2_or[:,0:aligned_map.shape[1]] = aligned_map
    packet2_or = packet2_or[real_data['shuffled_index'],0:21]
    

    #plot roc curve
    all_stim = np.concatenate((packet1.flatten(),packet2.flatten()))
    y_true = np.round(np.concatenate((packet1_or.flatten(),packet2_or.flatten())))
    fpr, tpr, _ = roc_curve(y_true, all_stim)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, all_stim, y_true

def supp_fig_triplet_corrs(folder):
    real_data = np.load(folder + '/triplet_corr_real.npz')['tr_corrs']
    spikeGAN = np.load(folder + '/triplet_corr_spikeGAN.npz')['tr_corrs']
    k_pairwise = np.load(folder + '/triplet_corr_k_pairwise.npz')['tr_corrs']
    DDG = np.load(folder + '/triplet_corr_DDG.npz')['tr_corrs']
    plt.figure(figsize=(10, 6),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    maximo = np.max(np.array([np.max(real_data),np.max(spikeGAN),np.max(k_pairwise)]))
    minimo = np.min(np.array([np.min(real_data),np.min(spikeGAN),np.min(k_pairwise)]))
   
    plt.plot([minimo,maximo],[minimo,maximo],colors[0])
    plt.plot(real_data,k_pairwise,'x'+colors[1])
    plt.plot(real_data,spikeGAN,'.'+colors[2])
    plt.plot(real_data,DDG,'+'+colors[3])
    plt.xlabel('triplet corrs expt')
    plt.ylabel('triplet corrs models')   
    plt.title('triplet correlations')
    
    plt.annotate('k-pairwise model',xy=(minimo,maximo-2*(maximo-minimo)/10),fontsize=8,color=colors[1])
    plt.annotate('DG model',xy=(minimo,maximo-3*(maximo-minimo)/10),fontsize=8,color=colors[3])
    plt.annotate('Spike-GAN',xy=(minimo,maximo-(maximo-minimo)/10),fontsize=8,color=colors[2])


def supp_fig_nearest_sample(num_neurons, num_bins, folder, num_cols=10, num_rows=10, main_folder=''):
    '''
    plots nearest sample for figure S5
    '''
    f = plt.figure(figsize=(8, 10),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    points = nearest_sample(num_neurons, num_bins, folder, 'spikeGAN', num_cols=10, num_rows=1, points=np.array([[0.01,0.095],0,0]), save_fig=False, f=f)
    
    points = nearest_sample(num_neurons, num_bins, folder, 'k_pairwise', num_cols=10, num_rows=1, points=np.array([[0.01,points[0,1]],0,0]), save_fig=False, f=f)
    
    points = nearest_sample(num_neurons, num_bins, folder, 'DDG', num_cols=10, num_rows=1, points=np.array([[0.01,points[0,1]],0,0]), save_fig=False, f=f)
    
    points = nearest_sample(num_neurons, num_bins, folder, 'real', num_cols=10, num_rows=1, points=np.array([[0.01,points[0,1]],0,0]), save_fig=False, f=f)
    
    f.savefig(folder+'nearest_sample_all_methods.svg',dpi=600, bbox_inches='tight')
    f.savefig(main_folder + '/figures paper/nearest_sample_all_methods.svg',dpi=600, bbox_inches='tight')
    plt.close(f)

def nearest_sample(num_neurons, num_bins, folder, name, num_cols=10, num_rows=10, points=np.array([[0.01,0.095],0,0]), save_fig=True, f=None, main_folder=''):
    #original_data = np.load(folder + '/stats_real.npz') 
    aux = np.load(folder+'/closest_sample_'+name+'.npz')
    closest_sample = aux['closest_sample']
    X = aux['samples']
    real_samples = np.load(folder+'/stats_real.npz')
    real_samples = real_samples['samples']
    
    if save_fig:
        f = plt.figure(figsize=(8, 10),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    num_samples = num_cols*num_rows
  
    width = 0.4
    height = width*num_bins/(3*num_neurons)
    margin = width/10
    factor_width = 0.35
    factor_height = -0.0025
    for i in range(num_samples):
        pos_h = (points[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)
        cbaxes = f.add_axes([pos_h, pos_v, width, height]) 
        sample = X[:,i]
        sample = sample.reshape(num_neurons,num_bins)
        
        cbaxes.imshow(sample,interpolation='nearest', cmap='gray')
        if i==int(num_cols/2):
            cbaxes.set_title(name[int(name=='DDG'):],fontsize=14)
        cbaxes.axis('off')      

        pos_h = (points[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)-height+factor_height
        cbaxes = f.add_axes([pos_h,pos_v, width, height]) 
        nearest_sample = real_samples[:,int(closest_sample[i])]
        nearest_sample = nearest_sample.reshape(num_neurons,num_bins)
        cbaxes.imshow(nearest_sample,interpolation='nearest', cmap='gray') 
        cbaxes.axis('off')  
    
    
    if save_fig:
        f.savefig(folder+name+'nearest_sample.svg',dpi=600, bbox_inches='tight')
        f.savefig(main_folder + '/figures paper/'+name+'nearest_sample.svg',dpi=600, bbox_inches='tight')
        plt.close(f)
    else:
        ax = plt.gca()
        points = ax.get_position().get_points()
        return points
    
    
    
def main(main_folder):
    plt.close('all')
    print('nearest sample fig')
    folder = main_folder + 'samples conv/dataset_retina_num_samples_8192_num_neurons_50_num_bins_32_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_21/'
    supp_fig_nearest_sample(num_neurons=50, num_bins=32, folder=folder, num_cols=10, num_rows=10, main_folder=main_folder)
#
#
    print('supp figure with negative corrs')
    dataset = 'uniform'
    num_samples = '8192'
    num_neurons = '16'
    num_bins = '128'
    ref_period = '2'
    firing_rate = '0.25'
    correlation = '0.3'
    group_size = '2'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '20'
    sample_dir = main_folder + 'samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir,folder_fc='', fig_2_or_3=3, neg_corrs=True, name='supp_negative_corrs', main_folder=main_folder)

    
   
    print('supp figure ROC curves')
    figure_supp_using_imp_maps_ROC_curves(main_folder=main_folder)
#    
    print('supp figure using importance maps with less samples')
    num_samples = 4096#1024#2048#8192#
    packet_noise = 0.0#0.5#
    folder = main_folder + 'samples conv/dataset_packets_num_samples_'+str(num_samples)+'_num_neurons_32_num_bins_32_packet_prob_1.0_firing_rate_0.05_group_size_18_noise_in_packet_'+str(packet_noise)+'_number_of_modes_2_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_20/'    
    
    num_samples = 2048#1024#2048#8192#
    packet_noise = 0.0#0.5#
    folder = main_folder + 'samples conv/dataset_packets_num_samples_'+str(num_samples)+'_num_neurons_32_num_bins_32_packet_prob_1.0_firing_rate_0.05_group_size_18_noise_in_packet_'+str(packet_noise)+'_number_of_modes_2_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_20/'    
    #num_samples = 8000
    figure_supp_using_imp_maps(folder,num_samples=num_samples, name='_packetNoise'+str(packet_noise)+'_II', noise=packet_noise, main_folder=main_folder)
    num_samples = 1024#1024#8192#4096#
    packet_noise = 0.0#0.5#
    folder = main_folder + 'samples conv/dataset_packets_num_samples_'+str(num_samples)+'_num_neurons_32_num_bins_32_packet_prob_1.0_firing_rate_0.05_group_size_18_noise_in_packet_'+str(packet_noise)+'_number_of_modes_2_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_20/'    
    #num_samples = 8000
    figure_supp_using_imp_maps(folder,num_samples=num_samples, name='_packetNoise'+str(packet_noise)+'_II', noise=packet_noise, main_folder=main_folder)
#    
    print('supp figure using importance maps with noisy packets')
    num_samples = 4096#1024#2048#8192#
    packet_noise = 0.5#0.5#
    folder = main_folder + 'samples conv/dataset_packets_num_samples_'+str(num_samples)+'_num_neurons_32_num_bins_32_packet_prob_1.0_firing_rate_0.05_group_size_18_noise_in_packet_'+str(packet_noise)+'_number_of_modes_2_critic_iters_5_lambda_10_num_layers_2_num_features_128_kernel_5_iteration_20/'    
#    
# 
    print('FIGURE 4')
    dataset = 'packets'
    num_samples = '8192'
    num_neurons = '32'
    num_bins = '64'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '21'
    packet_prob = '0.1'
    firing_rate = '0.1'
    group_size = '8'

    sample_dir = main_folder + 'samples conv/' + 'dataset_' + dataset + '_num_samples_' + str(num_samples) +\
          '_num_neurons_' + str(num_neurons) + '_num_bins_' + str(num_bins) + '_packet_prob_' + str(packet_prob)\
          + '_firing_rate_' + str(firing_rate) + '_group_size_' + str(group_size)  + '_critic_iters_' +\
          str(critic_iters) + '_lambda_' + str(lambd) +\
          '_num_layers_' + str(num_layers)  + '_num_features_' + str(num_features) + '_kernel_' + str(kernel) +\
          '_iteration_' + iteration + '/'
    figure_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, main_folder=main_folder)
    
    figure_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir,num_rows=5, main_folder=main_folder)
  
    
    print('FIGURE 3')
    dataset = 'retina'
    num_samples = '8192'
    num_neurons = '50'
    num_bins = '32'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '21'
    sample_dir = main_folder + 'samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir,folder_fc='', fig_2_or_3=3, name='fig_3_retinal_data', main_folder=main_folder)


    print('FIGURE 2')
    dataset = 'uniform'
    num_samples = '8192'
    num_neurons = '16'
    num_bins = '128'
    ref_period = '2'
    firing_rate = '0.25'
    correlation = '0.3'
    group_size = '2'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '20'
    num_units = '490'
    sample_dir = main_folder + 'samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    sample_dir_fc = main_folder + 'samples fc/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd + '_num_units_' + num_units +\
          '_iteration_' + iteration + '/'
          
    points_colorbar, cbaxes, map_aux, maximo, minimo= figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, folder_fc=sample_dir_fc, fig_2_or_3=2, name='fig_2_sims', main_folder=main_folder)
    
    

if __name__ == '__main__':
   main('/home/manuel/Spike-GAN/') 
    
 