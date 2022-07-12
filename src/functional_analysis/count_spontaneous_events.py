from cmath import sqrt
from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import sys
import json
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells
from scipy.stats import zscore

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
cell_dictionary_file = config['AnalysisFile']
cell_dictionary_file_out = cell_dictionary_file
EPOCH_START_IN_MS = config['EpochStart']
EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
FRAMERATE = config['RecordingFR']

def main():
    # for each cell
    # look at the 10 frames preceding the current frame
    # zscore in relation to these frames
    # if the zscore is more than 3?? z-scores, it is an event
    # may need to implement a skip-forward after event identification..
    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    
    pass

if __name__=="__main__":
    main()