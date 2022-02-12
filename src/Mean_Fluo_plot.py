# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:29:20 2021

@author: Conor Lane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os



#Path to recording folder containing sutie2p data
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
    plt.title('Mean Fluorescence, All Cells')
    
    return fig

make_figs(root)