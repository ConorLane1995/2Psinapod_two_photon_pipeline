"""
Script to plot single cells' average responses to each frequency presented
INPUT: cell_dictionary.pkl
AUTHOR: Veronica Tarka, May 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib.pyplot as plt
from seaborn import color_palette
import pickle
import random
import json
import os

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']
RECORDING_FRAMERATE = config['RecordingFR']
EPOCH_START_IN_MS = config['EpochStart']

N_CELL_SAMPLES = 20

"""
@param cell: the contents of the 'traces' key in the cell
@return return_d: a dictionary with a key for every frequency, where each key contains the average response to that frequency across all intensities
"""
def get_avg_traces(cell):

    # cell is going to be structured like this:
    # freq { intensity { repetition: trace }}}

    # let's return a dictionary structured like this:
    # freq : avg_trace

    return_d = dict.fromkeys(cell.keys())

    for freq in cell:

        for intensity in cell[freq]:
            sum_trace = 0
            num_rep = 0

            for rep in cell[freq][intensity]:
                sum_trace += cell[freq][intensity][rep]
                num_rep += 1

        avg_trace = sum_trace / num_rep
        return_d[freq] = avg_trace

    return return_d

"""
Plots subplot for each cell in the random sample (size specified by N_CELL_SAMPLES), where each subplot will show
the average response for that cell to each of the frequencies presented.
@param active_cells: the big cell dictionary but only with active cells
@param y_limit: how high to make the y-axis
"""
def plot_trials(active_cells,y_limit):
    # epoched traces is a dictionary structure as:
    # {cell : freq : intensity: repetition: trace}

    pal = color_palette('hls', len(active_cells[next(iter(active_cells))]['traces'].keys()))

    # make sure we are pulling an even number of cell samples
    n_cell_samples = N_CELL_SAMPLES
    if (n_cell_samples % 2 != 0):
        n_cell_samples += 1

    # figure out how many frames were pre-stimulus
    n_baseline_frames = int((EPOCH_START_IN_MS/-1000*RECORDING_FRAMERATE)-1)

    # get a random sampling of cells
    # sample_idx = random.sample(range(len(epoched_traces)),20)
    all_active_cells = list(active_cells)
    cell_sample = random.sample(all_active_cells,n_cell_samples)

    fig, axs = plt.subplots(nrows=int(n_cell_samples/2), ncols=2)
    axs = axs.ravel()

    for ax,cell,ctr in zip(axs,cell_sample,range(len(axs))):
        avg_traces = get_avg_traces(active_cells[cell]['traces'])

        len_trace = 0
        for trace,color in zip(avg_traces,pal):
            ax.plot(avg_traces[trace],c=color)
            len_trace = len(avg_traces[trace])

        if (ctr != len(axs)-2) and (ctr != len(axs)-1):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_xlabel("Time since stimulus onset (ms)")
            ax.set_xticks(range(0,len_trace,2))
            ax.set_xticklabels(range(0-n_baseline_frames*100,(len_trace-n_baseline_frames)*100,200))
            ax.set_ylabel("dF/F")

        ax.set_title(cell)

        # add a line to show exactly where stimulus happened
        # epoching started 100 ms before the trigger
        # so have an extra 0.1s * RECORDING_FRAMERATE frames before the trigger
        ax.axvline(x=n_baseline_frames)
        ax.autoscale(enable=True, axis='x', tight=True)
        
        # set the limits on the y axis
        ax.set_ylim([0,y_limit])
        ax.set_yticks([y_limit/2])
        
    fig.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.show(block=False)

"""
Takes the big dictionary, and returns a dictionary of the same format but only with active cells
@param cell_dict: the big cell dictionary
@return d: dictionary of active cells formatted identically to cell_dict
"""
def get_active_cells(cell_dict):

    d = dict.fromkeys(cell_dict.keys())

    for cell in cell_dict:
        if cell_dict[cell]['active'] == True:
            d[cell] = cell_dict[cell]
        else:
            d.pop(cell,None)

    return d

def main():

    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)
    
    active_cells = get_active_cells(cell_dict)
    plot_trials(active_cells,1000)
    plt.show()

if __name__=="__main__":
    main()