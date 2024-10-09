"""
Plots the tuning heatmap and tuning curve traces for a single cell, specified in CELL_OF_INTEREST
INPUT: cell_dictionary.pkl, recording_info.pkl
AUTHOR: Veronica Tarka, August 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']

CELL_OF_INTEREST = 809

"""
Shows the tuning heatmap for a single cell, specified by the CELL_OF_INTEREST ID number
@param cell_tuning: the contents of the 'tuning' key for the cell
@param cell_ID: the ID of the cell to be plotted
@param frequencies: a list of frequencies presented during the recording in ascending order
@param intensities: a list of intensities presented during the recording in ascending order
"""
def plot_single_tuning_curve(cell_tuning,cell_ID,frequencies,intensities):

    fig = plt.figure()
    ax = fig.gca()

    im = plt.imshow(np.transpose(cell_tuning),cmap='winter',origin='lower')
    plt.colorbar(im)

    if len(frequencies)>5: # if we have a lot of frequencies
        # only show a label for every second frequency
        ax.set_xticks(range(0,len(frequencies),2))
        ax.set_xticklabels(frequencies[range(0,len(frequencies),2)])
    else:
        # otherwise show every frequency label
        ax.set_xticks(range(0,len(frequencies)))
        ax.set_xticklabels(frequencies[range(0,len(frequencies))])

    # show every intensity label
    ax.set_yticks(range(0,len(intensities)))
    ax.set_yticklabels(intensities)

    # label the axes
    ax.set_ylabel("Intensity (dB)")
    ax.set_xlabel("Frequency (Hz)")

    plt.title(cell_ID)
    plt.show(block=False)

"""
Plot the tuning traces for a single cell
@param cell_traces: the contents of the 'traces' key for a single cell in the big dictionary
@param n_frequencies: the total number of unique frequencies presented during the recording
@param n_intensities: the total number of unique intensities presented during the recording
@param y_limit: how tall the y axis should be for each subplot
"""
def plot_tuning_traces(cell_traces,n_frequencies,n_intensities,y_limit):

    fig,axs = plt.subplots(n_intensities,n_frequencies,sharex='col',sharey='row',figsize=(14,5))

    for row,freq in zip(range(n_frequencies),cell_traces.keys()):
        for col,itsy in zip(range(n_intensities),reversed(list(cell_traces[freq].keys()))):
            for rep in cell_traces[freq][itsy]:
                axs[col,row].plot(cell_traces[freq][itsy][rep]) # plot every trial

            # miscellaneous formatting
            axs[col,row].set_xticks([])
            axs[col,row].set_yticks([])
            if row==0:
                axs[col,row].set_ylabel(itsy) # add the intensity to the far left edge
            if col==n_intensities-1:
                axs[col,row].set_xlabel(freq) # add the frequency at the bottom
            axs[col,row].axvline(x=4,color='k',linestyle='--')
            axs[col,row].set_ylim(bottom=0,top=y_limit)
            axs[col,row].autoscale(enable=True, axis='x', tight=True)

    fig.subplots_adjust(wspace=0,hspace=0)
    fig.text(0.5,0.01,"Frequency (Hz)",va='center',ha='center')
    fig.text(0.01,0.5,"Intensity (dB)",va='center',ha='center',rotation='vertical')
    fig.suptitle(CELL_OF_INTEREST)
    plt.show(block=False)


def main():

    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    # load the recording info file
    with open(BASE_PATH + "recording_info.pkl","rb") as f:
        recording_info = pickle.load(f)

    frequencies = recording_info['frequencies']
    intensities = recording_info['intensities']

    plot_single_tuning_curve(cell_dict[CELL_OF_INTEREST]['tuning'],CELL_OF_INTEREST,frequencies,intensities)
    plot_tuning_traces(cell_dict[CELL_OF_INTEREST]['traces'],len(frequencies),len(intensities),100)
    plt.show()

if __name__=="__main__":
    main()