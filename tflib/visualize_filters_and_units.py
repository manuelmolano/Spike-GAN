#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:20:24 2017

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import tensorflow as tf
#import time

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

def plot_filters(filters, sess, config, index):
    for ind_layer in range(1):#len(filters)):
        filter_temp = filters[ind_layer].eval(session=sess)
        filter_aux = filter_temp[:,:,0]
        filter_aux = filter_aux[:,:].T
        
        num_filters = filter_temp.shape[2]
        num_cols = int(np.ceil(np.sqrt(num_filters)))
        #print(np.corrcoef(all_filters.T))
        
        height = 1/16
        width = 1/16
        factor = 0.052
        f = plt.figure(figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        for i in range(num_filters):
            cbaxes = f.add_axes([0.01+(i%num_cols)*(width-factor), 0.97-height*np.floor(i/num_cols), width, height]) 
            filter_aux = filter_temp[:,:,i]
            filter_aux = filter_aux[:,:].T
            filter_aux = filter_aux[index,:]
            cbaxes.imshow(filter_aux,interpolation='nearest',cmap='gray')#,clim=[0,np.max(filter_temp.flatten())]
            cbaxes.axis('off')  
          
          
        f.savefig(config.sample_dir+'filters_layer_' + str(ind_layer) + '.svg',dpi=600, bbox_inches='tight')
        plt.close(f)  
        

def plot_untis_rf_conv(activations, outputs, inputs, sess, config, index):
    my_cmap = plt.cm.gray
    num_layers = len(activations)
    critics_decision = outputs.eval(session=sess)
    inputs = inputs.eval(session=sess)
    critics_decision = critics_decision.reshape(1,len(critics_decision))
    num_rows = int(np.ceil(np.sqrt(num_layers)))
    num_cols = int(np.ceil(np.sqrt(num_layers)))
    f,sbplt = plt.subplots(num_rows,num_cols,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
   
    for ind_f in range(num_layers):
        act_temp = activations[ind_f].eval(session=sess)  
        act_shape = act_temp.shape
        num_features = act_shape[1]
        #num_bins = act_shape[2]
        for ind_bin in range(1,2):
            corr_with_decision = np.zeros((num_features,))
            #compute correlation between units and final decision
            for ind_feature in range(num_features):
                    act_aux = act_temp[:,ind_feature,ind_bin].reshape(1,act_temp.shape[0])
                    aux = np.corrcoef(np.concatenate((act_aux,critics_decision),axis=0))
                    corr_with_decision[ind_feature] = abs(aux[1,0])
                
            sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].hist(corr_with_decision)  
            sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].set_title(str(np.mean(np.abs(corr_with_decision))))
            
            f2,sbplt2 = plt.subplots(10,10,figsize=(8, 8),dpi=250)
            matplotlib.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
            #get STA for most correlated units
            portion = np.min([100/num_features,1])
            most_corr = corr_with_decision>np.percentile(corr_with_decision,100-portion*100)
            most_corr = act_temp[:,most_corr,ind_bin]
            for ind_unit in range(most_corr.shape[1]):
                act_aux = most_corr[:,ind_unit].reshape(act_temp.shape[0],1)
                spk = np.sum(inputs*act_aux,axis=0)
                spk = spk.reshape(config.num_neurons,config.num_bins)
                spk = spk[index,:]
                sbplt2[int(np.floor(ind_unit/10))][ind_unit%10].imshow(spk,interpolation='nearest', cmap = my_cmap)#, clim=(np.min(most_corr.flatten()), np.max(most_corr.flatten())))  
                sbplt2[int(np.floor(ind_unit/10))][ind_unit%10].axis('off')  
            print(config.sample_dir+'sta_layer_' + str(ind_f) + '_bin_' + str(ind_bin) + '_most_corr.svg')  
            f2.savefig(config.sample_dir+'sta_layer_' + str(ind_f) + '_bin_' + str(ind_bin) + '_most_corr.svg',dpi=600, bbox_inches='tight')
            plt.close(f2)  
            
        
        
    f.savefig(config.sample_dir+'correlations.svg',dpi=600, bbox_inches='tight')
    plt.close(f)  
    
    

def plot_untis_rf(activations, outputs, inputs, sess, config, index):
    my_cmap = plt.cm.gray
    num_layers = len(activations)
    critics_decision = outputs.eval(session=sess)
    inputs = inputs.eval(session=sess)
    critics_decision = critics_decision.reshape(1,len(critics_decision))
    num_rows = int(np.ceil(np.sqrt(num_layers)))
    num_cols = int(np.ceil(np.sqrt(num_layers)))
    f,sbplt = plt.subplots(num_rows,num_cols,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
   
    for ind_f in range(num_layers):
        act_temp = activations[ind_f].eval(session=sess) 
        act_shape = act_temp.shape
        num_units = act_shape[1]
        corr_with_decision = np.zeros((num_units,))
        counter = 0
        #compute correlation between units and final decision
        for ind_unit in range(num_units):
            act_aux = act_temp[:,ind_unit].reshape(1,act_temp.shape[0])
            aux = np.corrcoef(np.concatenate((act_aux,critics_decision),axis=0))
            corr_with_decision[counter] = abs(aux[1,0])
            counter += 1
        sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].hist(corr_with_decision)  
        sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].set_title('L'+ str(ind_f) +' '+ str('{0:.2f}'.format(np.mean(np.abs(corr_with_decision)))))
        
        f2,sbplt2 = plt.subplots(10,10,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
   
        #get STA for most correlated
        portion = np.min([100/num_units,1])
        most_corr = corr_with_decision>np.percentile(corr_with_decision,100-portion*100)
        most_corr = act_temp[:,most_corr]
        for ind_unit in range(most_corr.shape[1]):
            act_aux = most_corr[:,ind_unit].reshape(act_temp.shape[0],1)
            spk = np.sum(inputs*act_aux,axis=0)
            spk = spk.reshape(config.num_neurons,config.num_bins)
            spk = spk[index,:]
            sbplt2[int(np.floor(ind_unit/10))][ind_unit%10].imshow(spk,interpolation='nearest', cmap = my_cmap)#, clim=(np.min(most_corr.flatten()), np.max(most_corr.flatten())))  
            sbplt2[int(np.floor(ind_unit/10))][ind_unit%10].axis('off')  
        f2.savefig(config.sample_dir+'sta_layer_' + str(ind_f) + '_most_corr.svg',dpi=600, bbox_inches='tight')
        plt.close(f2)  
        
        
        
    f.savefig(config.sample_dir+'correlations.svg',dpi=600, bbox_inches='tight')
    plt.close(f)  


def plot_histogram(values, folder, name):
    f,ax = plt.subplots(1,1,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
    plt.hist(values)
    f.savefig(folder + name + 'critics_output.svg',dpi=600, bbox_inches='tight')
    plt.close(f)  
    