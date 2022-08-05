"""
Script to generate the files useful for performing a tSNE
INPUT: cell_dictionary.pkl, recording_info.pkl, raw traces (nCells x nFrames), epoched traces (nCells x nTrials x nFrames)
OUTPUT: BF_labeling (list of best frequencies for all active cells), raw traces and epoched traces for only active cells
AUTHOR: Veronica Tarka, July 2022, veronica.tarka@mail.mcgill.ca
"""

import numpy as np
import pickle
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)
 
BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']

"""
Determine a cell's best frequency from the tuning matrix stored in the dictionary
@param cell_tuning: an array of the cell's responses to the stimuli arranged so that axis 0 are the frequencies and axis 1 are the intensities
@param freqs: a list of the unique frequencies presented during the recording, arranged so freqs[0] corresponds with the responses in cell_tuning[0,:]
@return best frequency as calculated by the frequency eliciting the largest median response across all intensities
"""
def get_BF(cell_tuning,freqs):
    median_across_itsies = np.median(cell_tuning, axis=1) # determine the median response to the frequencies across all intensities
    max_response_idx = np.argmax(median_across_itsies) # find which frequency elicited the strongest median response
    return freqs[max_response_idx]


def main():
    # load the epoched traces NOT in dictionary form (part of output from epoch_recording.py)
    raw_traces = np.load(BASE_PATH+"raw_corrected_traces_all.npy")
    epoched_traces = np.load(BASE_PATH+"epoched_traces_all.npy")

    # load the dictionary file with the epoched traces and tuning info
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dictionary = pickle.load(f)

    # load the recording/stimuli information
    with open(BASE_PATH+"recording_info.pkl","rb") as f:
        recording_info = pickle.load(f)

    # get only the cells we've considered active (dictionary should have 'active' key holding boolean)
    active_cell_dictionary = get_active_cells(cell_dictionary)

    # make a list of all the IDs of the active cells
    active_cells = np.array(list(active_cell_dictionary.keys()))
    active_cells_idx = active_cells - 1

    # extract the raw traces and epoched traces for only the active cells
    active_raw_traces = raw_traces[active_cells_idx,:]
    active_epoched_traces = epoched_traces[active_cells_idx,:,:]

    # make a list of the best frequencies for each active cell (in order)
    frequencies = recording_info['frequencies']
    BFs = []
    for cell in active_cell_dictionary:
        BFs.append(get_BF(active_cell_dictionary[cell]['tuning'],frequencies))

    np.save(BASE_PATH+"BF_labeling.npy",BFs) # a list of the active cells' BFs
    np.save(BASE_PATH+"active_corrected_traces.npy",active_raw_traces) # save the trace for each active cell
    np.save(BASE_PATH+"active_epoched_traces.npy",active_epoched_traces) # the trace for the trials

if __name__ == "__main__":
    main()