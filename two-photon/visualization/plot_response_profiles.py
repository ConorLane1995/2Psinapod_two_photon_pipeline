import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from cmath import sqrt
import os
import sys
import json

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

    
BASE_PATH = config['RecordingFolder']
cell_dictionary_file = config['AnalysisFile']
cell_dictionary_file_out = cell_dictionary_file
EPOCH_START_IN_MS = config['EpochStart']
EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
FRAMERATE = config['RecordingFR']

def get_profile(cell_tuning):
    tmp = np.average(cell_tuning, axis=1)
    return list(tmp)

def get_best_frequency(cell_tuning,freqs):
    median_across_itsies = np.median(cell_tuning, axis=1)
    max_response_idx = np.argmax(median_across_itsies)
    return freqs[max_response_idx]

def main():
    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    freqs = [4.4,5.4,6.6,8.1,10,12,15,19,23,28,28,35,43]

    active_profiles = []
    inactive_profiles = []

    active_BFs = []
    inactive_BFs = []
    # for each neuron
    for cell in cell_dictionary:
        # check if the cell is active

        # if active, add row to active list
        if cell_dictionary[cell]['active'] == True:
            active_profiles.append(get_profile(cell_dictionary[cell]['tuning_curve_peak']))
            active_BFs.append(get_best_frequency(cell_dictionary[cell]['tuning_curve_peak'],freqs))

        # if not active, add row to inactive list
        else:
            inactive_profiles.append(get_profile(cell_dictionary[cell]['tuning_curve_peak']))
            inactive_BFs.append(get_best_frequency(cell_dictionary[cell]['tuning_curve_peak'],freqs))



    # sort the neurons by best frequency
    sorted_active_profiles = [x for y,x in sorted(zip(active_BFs,active_profiles))]
    sorted_inactive_profiles = [x for y,x in sorted(zip(inactive_BFs,inactive_profiles))]

    full_profiles = sorted_active_profiles + sorted_inactive_profiles

    for row in full_profiles: # in range(len(inactive_profiles)): 
        if len(row)!= 12:
            print("Found problem")
        print(row[0])

    plt.show()

    plt.imshow(full_profiles,extent=[0,5,0,10],aspect=1.1,cmap=plt.cm.magma)
    plt.axhline(y=10-len(active_BFs)/38.5,color='white')
    plt.show()


if __name__ == "__main__":
    main()