"""
TODO
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../../')
from utils import get_active_cells

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']
CELL_DICT_FILE_OUT = CELL_DICT_FILE
EPOCH_START_IN_MS = config['EpochStart']
FRAMERATE = config['RecordingFR']

"""
Show a plot containing the tuning heatmaps for the first 25 active cells in this recording, each subplot title is the cell ID
@param cell_dictionary: the big cell dictionar, with key 'tuning' containing the tuning curves
@param frequencies: each frequency that was presented during the recording in ascending order
@param intensities: each intensity that was presented during the recording in ascending order
"""
def plot_tuning_curves(cell_dictionary,frequencies,intensities):
    frequencies = np.round(frequencies / 1000) # convert to kHz
    frequencies = np.array([int(f) for f in frequencies]) # round to integer

    # create the figure
    fig,axs = plt.subplots(5,5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.5,wspace=0.001)
    axs = axs.ravel() # make the axis indexing a vector rather than 2D array
    for ax,cell in zip(axs,cell_dictionary.keys()):
        cell_tuning = cell_dictionary[cell]['tuning'] # get the tuning for this cell
        im = ax.imshow(np.transpose(cell_tuning),cmap='winter',origin='lower') # plot it
        ax.set_xticks([]) # hide the axis ticks
        ax.set_yticks([])
        ax.set_title(cell) # show the cell ID as the title
        plt.colorbar(im,ax=ax) # add the color bar

    if len(frequencies)>5: # if we have a lot of frequencies
        # only show a label for every second frequency
        axs[20].set_xticks(range(0,len(frequencies),2))
        axs[20].set_xticklabels(frequencies[range(0,len(frequencies),2)])
    else:
        # otherwise show every frequency label
        axs[20].set_xticks(range(0,len(frequencies)))
        axs[20].set_xticklabels(frequencies[range(0,len(frequencies))])

    # show every intensity label
    axs[20].set_yticks(range(0,len(intensities)))
    axs[20].set_yticklabels(intensities)

    # label the axes for just one subplot
    axs[20].set_ylabel("Intensity (dB)")
    axs[20].set_xlabel("Frequency (kHz)")

    plt.show(block=False)

def main():
        # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    # load the recording info file
    with open(BASE_PATH + "recording_info.pkl","rb") as f:
        recording_info = pickle.load(f)

    active_cell_dict = get_active_cells(cell_dict)
    
    frequencies = recording_info['frequencies']
    intensities = recording_info['intensities']

    plot_tuning_curves(active_cell_dict,frequencies,intensities)
    plt.show()

if __name__=="__main__":
    main()