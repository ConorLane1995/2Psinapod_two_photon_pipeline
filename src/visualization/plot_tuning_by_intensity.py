import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from cmath import sqrt
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '..\..\\')
from utils import get_active_cells
from scipy.stats import zscore

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'\..\..\config.json','r') as f:
    config = json.load(f)

    
BASE_PATH = config['RecordingFolder']
cell_dictionary_file = config['AnalysisFile']
cell_dictionary_file_out = cell_dictionary_file
EPOCH_START_IN_MS = config['EpochStart']
EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
FRAMERATE = config['RecordingFR']

CELL_OF_INTEREST = 37


def get_cell_tuning_by_peak(cell_traces,plot_TF):

    # a = np.empty((7,9,5,30), int)

    fig,axs = plt.subplots(7)

    # frequency_counter = 0
    for freq in cell_traces:

        intensity_counter = 6
        for intensity in cell_traces[freq]:
            
            # rep_counter = 0
            for rep in cell_traces[freq][intensity]:
                # print(cell_traces[freq][intensity][rep].shape)
                # print(a[intensity_counter,frequency_counter,rep,:].shape)
                # if rep < 5:
                #     a[intensity_counter,frequency_counter,rep,:] = cell_traces[freq][intensity][rep]
                #     rep += 1
                
                # print(intensity_counter)
                # print(intensity)
                axs[intensity_counter].plot(cell_traces[freq][intensity][rep])
                axs[intensity_counter].autoscale(enable=True, axis='x', tight=True)
                axs[intensity_counter].set_ylim(bottom=0,top=550)
                axs[intensity_counter].axvline(x=5,linestyle='dashed',color='black')
                axs[intensity_counter].xaxis.set_visible(False)
                axs[intensity_counter].yaxis.set_visible(False)
                axs[intensity_counter].text(29,0,intensity)

            intensity_counter -= 1

    axs[6].xaxis.set_visible(True)
    axs[6].yaxis.set_visible(True)
    axs[6].set_xlabel("Trial frame (10 frames / second)")
    axs[6].set_ylabel("dF/F")
    plt.show()

def main():
    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    get_cell_tuning_by_peak(cell_dictionary[CELL_OF_INTEREST]['traces'],True)

if __name__=="__main__":
    main()