# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os



#Path to recording folder
root = 'C:/Users/Conor/Documents/Imaging_Data/Two-Photon/Psilocybin Project/30112021_GCaMP6s_CL33/TSeries-11302021-1610-109_30Hz'


def make_figs(root):

    #Find path to suite2p output
    s2p_output = os.path.join(root,'suite2p\\plane0')
    
    #Fluo signal from all ROIs
    F = np.load(os.path.join(s2p_output,'F.npy'))
    
    #Neuropil signal
    Fneu = np.load(os.path.join(s2p_output,'Fneu.npy'))
    
    #Classification of ROI - is it a cell?
    iscell = np.load(os.path.join(s2p_output,'iscell.npy'))

    # Remove neuropil fluorescence from trace
    F_corrected = F - 0.7*Fneu


    #Select only ROI's identified as cells
    Fcell = F_corrected[iscell[:,0].astype('bool')]

    #Normalize by converting to zscore
    Fnorm = zscore(Fcell,axis=1)

    #Generate ad plot mean of all cell traces
    Fmean = np.mean(Fnorm.T,axis=1)
    fig = plt.plot(Fmean)
    plt.xlabel('Frames')
    plt.ylabel('z-score')
    
    
    return fig

make_figs(root)


#Load the necessary .npy files from suite2P output
#F = np.load('C:/Users/Conor/Documents/Imaging_Data/Two-Photon/Psilocybin Project/30112021_GCaMP6s_CL33/TSeries-11302021-1610-109_30Hz/suite2p/plane0/F.npy')
#Fneu = np.load('C:/Users/Conor/Documents/Imaging_Data/Two-Photon/Psilocybin Project/30112021_GCaMP6s_CL33/TSeries-11302021-1610-109_30Hz/suite2p/plane0/Fneu.npy')
#iscell = np.load('C:/Users/Conor/Documents/Imaging_Data/Two-Photon/Psilocybin Project/30112021_GCaMP6s_CL33/TSeries-11302021-1610-109_30Hz/suite2p/plane0/iscell.npy')





