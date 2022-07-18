import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from cmath import sqrt
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells

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
    raw_traces = np.load(BASE_PATH+"raw_corrected_traces_all.npy")
    epoched_traces = np.load(BASE_PATH+"epoched_traces_all.npy")

    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    active_cell_dictionary = get_active_cells(cell_dictionary)

    active_cells = np.array(list(active_cell_dictionary.keys()))
    active_cells_idx = active_cells - 1

    active_raw_traces = raw_traces[active_cells_idx,:]
    active_epoched_traces = epoched_traces[active_cells_idx,:,:]

    np.save(BASE_PATH+"active_corrected_traces.npy",active_raw_traces) # save the trace for each cell ROI 
    np.save(BASE_PATH+"active_epoched_traces.npy",active_epoched_traces) # save the trace for trial before it's formatted into a dictionary

if __name__ == "__main__":
    main()