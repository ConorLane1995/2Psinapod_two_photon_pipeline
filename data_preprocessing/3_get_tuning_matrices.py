"""
Script to estimate each cells' response to each frequency x intensity combination through three different methods (peak, z-score, avging)
Adds the tuning estimates into the big dictionary under the key 'tuning'
INPUT: cell_dictionary.pkl, recording_info.pkl
OUTPUT: cell_dictionary.pkl now with a key 'tuning' with response estimates to the stim types
AUTHOR: Conor Lane, Veronica Tarka, May 2022, conor.lane1995@gmail.com
"""
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import pickle
import os

# Import general functions from preprocess_utils file.
from preprocess_utils import load_config_from_json

# Create a class to extract the required elements from the config.json that are used here, and load as config. 
@dataclass
class Config:
    RecordingFolder: str
    AnalysisFile: str
    RecordingFR: int
    EpochStart: int
    ZscoreThreshold: int

# Load the required variables from config.json into the config class, only if they are listed in the class.
config = load_config_from_json(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json')), Config)

BASE_PATH = config.RecordingFolder
CELL_DICT_FILE = config.AnalysisFile
CELL_DICT_FILE_OUT = CELL_DICT_FILE
EPOCH_START_IN_MS = config.EpochStart
FRAMERATE = config.RecordingFR
ZSCORE_THRESHOLD = config.ZscoreThreshold
n_baseline_frames =  round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1


# FUNCTIONS:

def get_zscored_response(trial,baselines,n_baseline_frames):
    
    """
    Compute the z-scored response for a given trial based on all baselines in recording.
    INPUTS:
    trial (np.ndarray): The trial data.
    baselines (np.ndarray): The baseline values for z-scoring.
    n_baseline_frames (int): The number of frames to exclude for z-scoring.
    OUTPUT:
        np.ndarray: The z-scored response for the trial.
    """

    response = trial[n_baseline_frames:]

    baseline_mean = np.mean(baselines)
    baseline_std = np.std(baselines)

     # Calculate the z-scores using vectorized operations
    zscore_response = (response - baseline_mean) / baseline_std

    return zscore_response



def get_all_trial_baselines(cell_trace, n_baseline_frames):

    """
    Extracts and averages baseline activity from cell traces.

    INPUTS:
    cell_trace (dict): Nested dictionary of cell traces by frequency and intensity.
    n_baseline_frames (int): Number of frames for baseline calculation.

    OUTPUT:
    np.ndarray: Mean baseline activity for each frequency and intensity combination.
    """
     
    # Extract the number of frequencies, intensities, and repetitions
    nfreq = len(cell_trace)
    nInt = len(cell_trace[next(iter(cell_trace))])  # next(iter) gets you the first cell so you don't have to hard code 0.
    nrep = len(cell_trace[next(iter(cell_trace))][next(iter(cell_trace[next(iter(cell_trace))]))])
    
    # Create a NumPy array to hold trials
    trials = np.empty((nfreq, nInt, nrep, n_baseline_frames))

    # Populate the trials array
    for i, freq in enumerate(cell_trace):
        for j, intensity in enumerate(cell_trace[freq]):
            for k, rep in enumerate(cell_trace[freq][intensity]):
                trials[i, j, k] = rep

    # Extract baseline frames for each trial
    baselines = trials[:, :, :, :n_baseline_frames]
    
    # Individual mean value for each baseline, to feed into zscore function. 
    baselines = np.mean(baselines, axis=-1)

    return baselines




def get_cell_tuning_by_mean_zscore(cell_traces,baselines,n_baseline_frames):
    
    nFrequencies = len(cell_traces)
    nIntensities = len(cell_traces[next(iter(cell_traces))])
    tuning_curve = np.empty((nFrequencies, nIntensities))


    for frequency_counter, freq in enumerate(cell_traces):
        for intensity_counter, intensity in enumerate(cell_traces[freq]):
            all_trials_of_this_intensity = []

            # Iterate through each trial for the given frequency/intensity combo
            for trial in cell_traces[freq][intensity]:
                trace = cell_traces[freq][intensity][trial]
                zscore_response = get_zscored_response(trace, baselines, n_baseline_frames)
                all_trials_of_this_intensity.append(zscore_response)

            # Convert trials to a NumPy array and calculate the mean
            all_trials_as_np = np.array(all_trials_of_this_intensity)
            mean_response = np.mean(np.mean(all_trials_as_np, axis=0))

            # Store the mean response in the tuning curve
            tuning_curve[frequency_counter, intensity_counter] = mean_response

    return tuning_curve



def get_cell_tuning_by_peak_zscore(cell_traces,baselines,n_baseline_frames):
    
    nFrequencies = len(cell_traces)
    nIntensities = len(cell_traces[next(iter(cell_traces))])
    tuning_curve = np.empty((nFrequencies, nIntensities))


    for frequency_counter, freq in enumerate(cell_traces):
        for intensity_counter, intensity in enumerate(cell_traces[freq]):
            all_trials_of_this_intensity = []

            # Iterate through each trial for the given frequency/intensity combo
            for trial in cell_traces[freq][intensity]:
                trace = cell_traces[freq][intensity][trial]
                zscore_response = get_zscored_response(trace, baselines, n_baseline_frames)
                all_trials_of_this_intensity.append(zscore_response)

            # Convert trials to a NumPy array and calculate the mean
            all_trials_as_np = np.array(all_trials_of_this_intensity)
            mean_response = np.max(np.mean(all_trials_as_np, axis=0))

            # Store the mean response in the tuning curve
            tuning_curve[frequency_counter, intensity_counter] = mean_response

    return tuning_curve


def get_all_tuning_curves(cell_dictionary):

    """
    Updates each cell in the dictionary to include tuning curves based on z-scored mean and peak trial-averaged activity.

    INPUT:
    cell_dictionary (dict): Dictionary where each key is a cell identifier.

    OUTPUT:
    dict: Updated dictionary with 'tuning' and 'peak_tuning' for each cell.
    """
    # Iterate over each cell and compute the baseline once, then use it for both tuning metrics.
    for cell, data in cell_dictionary.items():
        deconvolved_traces = data['deconvolved_traces']
        baselines = get_all_trial_baselines(deconvolved_traces, n_baseline_frames)

        # Add the tuning curves directly using functions.
        cell_dictionary[cell].update({
            'tuning': get_cell_tuning_by_mean_zscore(deconvolved_traces, baselines,n_baseline_frames),
            'peak_tuning': get_cell_tuning_by_peak_zscore(deconvolved_traces, baselines,n_baseline_frames)
        })

    return cell_dictionary



def main():

    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dictionary = pickle.load(f)
    
    cell_dictionary_with_tuning = get_all_tuning_curves(cell_dictionary) # add the key 'tuning' to the dictionary

    
    # save the edited dictionary
    with open(BASE_PATH+CELL_DICT_FILE_OUT,'wb') as f:
        pickle.dump(cell_dictionary_with_tuning,f)

    
    print("Tuning extraction complete! To see tuning for individual cells, go to plot_single_cell_tuning in rapid_visualization_packages")


if __name__ == '__main__':
    main()

