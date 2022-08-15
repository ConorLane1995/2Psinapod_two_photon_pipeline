"""
Plots response profiles as shown in Figure 1 of this paper: https://elifesciences.org/articles/53462
INPUT: cell_dictionary (with 'tuning' key) and recording_info.pkl
AUTHOR: Veronica Tarka, July 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pickle
import json
import os

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']

"""
Function to return the tuning curves averaged over intensities
@param cell_tuning: nItsy x nFreq array containing the response values for each itsy and freq combination (contents of 'tuning' key)
@return nFreq x 1 array containing response values averaged over intensities for every frequency
"""
def get_profile(cell_tuning):
    tmp = np.average(cell_tuning, axis=1)
    return list(tmp)

"""
Function to find the best frequency of a neuron based on the freq that elicited the strongest median response across all intensities.
@param cell_tuning: nItsy x nFreq array containing the response values for each itsy and freq combination (contents of 'tuning' key)
@param freqs: a list of frequencies presented in the recording in ascending order
@return the frequency that elicited the strongest median response across intensities
"""
def get_best_frequency(cell_tuning,freqs):
    median_across_itsies = np.median(cell_tuning, axis=1)
    max_response_idx = np.argmax(median_across_itsies)
    return freqs[max_response_idx]

"""
Function to resize an imshow figure so the pixels aren't square
Borrowed from here: https://moonbooks.org/Articles/How-to-change-imshow-aspect-ratio-in-matplotlib-/
"""
def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def main():
    # load big dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dictionary = pickle.load(f)

    # load our recording info file
    with open(BASE_PATH + "recording_info.pkl", 'rb') as f:
        recording_info = pickle.load(f)

    # get the frequencies we presented and convert it to kHz
    freqs = recording_info['frequencies']/1000

    # make empty lists we'll append cells into
    active_profiles = []
    inactive_profiles = []

    active_BFs = []
    inactive_BFs = []

    # for each neuron
    for cell in cell_dictionary:
        # check if the cell is active
        # if active, add row to active list
        if cell_dictionary[cell]['active'] == True:
            active_profiles.append(get_profile(cell_dictionary[cell]['tuning']))
            active_BFs.append(get_best_frequency(cell_dictionary[cell]['tuning'],freqs))

        # if not active, add row to inactive list
        else:
            inactive_profiles.append(get_profile(cell_dictionary[cell]['tuning']))
            inactive_BFs.append(get_best_frequency(cell_dictionary[cell]['tuning'],freqs))

    # sort the neurons by best frequency
    sorted_active_profiles = [x for _,x in sorted(zip(active_BFs,active_profiles))]
    sorted_inactive_profiles = [x for _,x in sorted(zip(inactive_BFs,inactive_profiles))]

    # concatenate the active and inactive into one list
    full_profiles = sorted_active_profiles + sorted_inactive_profiles

    # make our figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(full_profiles,cmap=plt.cm.magma) # plot the profiles
    forceAspect(ax,aspect=0.5) # resize so the pixels aren't square (looks terrible that way)
    ax.axhline(y=len(active_BFs),color='white') # add a line to divide the active and inactive cells

    # label every other frequency
    ax.set_xticks(range(0,len(freqs),2)) 
    ax.set_xticklabels(["{:4.1f}".format(i) for i in freqs[range(0,len(freqs),2)]])

    # add a color bar showing the range of values we're looking at
    cbar = fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(min(full_profiles)),vmax=max(max(full_profiles))),cmap=plt.cm.magma))
    
    # label everything
    cbar.set_label("Z-Scores from baseline (avgd over all intensities)")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Cells")
    plt.title("Active cells (above white line) and inactive cells sorted by BF")
    plt.show()


if __name__ == "__main__":
    main()