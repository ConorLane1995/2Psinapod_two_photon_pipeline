"""
Plots the tuning heatmaps (nIntensities x nFrequencies) for every active cell
INPUT: cell_dictionary.pkl, recording_info.pkl
AUTHOR: Veronica Tarka, August 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
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

"""
Show a plot containing the tuning heatmaps for the first 25 active cells in this recording, each subplot title is the cell ID
@param cell_dictionary: the big cell dictionar, with key 'tuning' containing the tuning curves
@param frequencies: each frequency that was presented during the recording in ascending order
@param intensities: each intensity that was presented during the recording in ascending order
"""
def plot_tuning_curves(cell_dictionary,frequencies,intensities):
    frequencies = np.round(frequencies / 1000) # convert to kHz
    frequencies = np.array([int(f) for f in frequencies]) # round to integer

    cell_IDs = list(cell_dictionary.keys()) # get a list of all the active cells
    nfigs = math.ceil(len(cell_dictionary) / 25) # figure out how many figures we need (25 cells per figure)

    for f in range(nfigs):
        # get our subset of keys
        cell_sample = cell_IDs[25*f:25*(f+1)]

        # create the figure
        fig,axs = plt.subplots(5,5,figsize=(20,13))
        fig.subplots_adjust(hspace=0.5,wspace=0.001)
        axs = axs.ravel() # make the axis indexing a vector rather than 2D array
        for ax,cell in zip(axs,cell_sample):
            cell_tuning = cell_dictionary[cell]['tuning'] # get the tuning for this cell
            im = ax.imshow(np.transpose(cell_tuning),cmap='winter',origin='lower') # plot it
            ax.set_xticks([]) # hide the axis ticks
            ax.set_yticks([])
            ax.set_title(cell) # show the cell ID as the title
            plt.colorbar(im,ax=ax,format=lambda x, _:f"{x:4.0f}") # add the color bar

        # if we have unfilled subplots, hide them
        if len(cell_sample) < 25:
            for ax_ctr in range(24,len(cell_sample)-1,-1):
                axs[ax_ctr].axis('off')

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

        plt.savefig(BASE_PATH+"tuning{}.png".format(f))
        plt.show(block=False)

def main():
    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    # load the recording info file
    with open(BASE_PATH + "recording_info.pkl","rb") as f:
        recording_info = pickle.load(f)

    active_cell_dict = get_active_cells(cell_dict) # get the active cells
    
    # TODO sort this out so the user can choose one of two options
    #active_cells = np.load(BASE_PATH + "active_cells.npy")
    #active_cell_dict = dict((k, cell_dict[k]) for k in active_cells)
    frequencies = recording_info['frequencies']
    intensities = recording_info['intensities']

    plot_tuning_curves(active_cell_dict,frequencies,intensities)
    plt.show()

if __name__=="__main__":
    main()