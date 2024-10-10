"""
Script to take extracted calcium traces from Suite2P and epoch the recording into trials, organized by stimulus parameters.
INPUT: stim triggers in csv, Suite2P outputs (F.npy, Fneu.npy, iscell.npy)
OUTPUT: epoched_F.npy formatted as nCells x nTrials x nFrames array
        onsets.npy - list of frames where triggers occured
        raw_corrected_traces.npy - nNeurons x nFrames fluorescence traces (not epoched)
        cell_dictionary (.pkl) - dictionary of each cell ROI with epoched traces stored in {'traces' {freq {intensity {repetition}}}}
AUTHORS: Conor Lane, Veronica Tarka, January 2022, conor.lane1995@gmail.com
"""

import time
start_time = time.time()
from dataclasses import dataclass
import os
import numpy as np
import scipy.io as scio
import pickle

# Import general functions from preprocess_utils file.
from preprocess_utils import load_config_from_json

# Create a class to extract the required elements from the config.json that are used here, and load as config. 
@dataclass
class Config:
    RecordingFolder: str
    Triggers: str
    Conditions: str
    AnalysisFile: str
    TriggerFR: int
    TriggerDelay: int
    RecordingFR: int
    EpochStart: int
    EpochEnd: int
    Neuropil: int

# Load the required variables from config.json into the config class, only if they are listed in the class.
config = load_config_from_json(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json')), Config)

# Extract filepaths and recording parameters from the config class. 
BASE_PATH = config.RecordingFolder
CSV_PATH = config.Triggers
CONDITIONS_PATH = config.Conditions
OUTPUT_PATH = config.AnalysisFile
STIMULUS_FRAMERATE = config.TriggerFR
TRIGGER_DELAY_IN_MS = config.TriggerDelay
RECORDING_FRAMERATE = config.RecordingFR
EPOCH_START_IN_MS = config.EpochStart
EPOCH_END_IN_MS = config.EpochEnd
NEUROPIL_CORRECTION = config.Neuropil


def get_stimulus_onset_frames(stimulus):

    """
    Find the stimulus onsets from the triggers CSV and define as frames in the fluorescence recording
    @param stimulus: 2D vector of the voltage trace of the stimulus triggers
    @return onset_frames: a list of the frames in the fluo recording where the stim was presented
    """

    stimulus = np.array(stimulus)

    # Get the length of a single trial epoch
    trial_length = (EPOCH_END_IN_MS - EPOCH_START_IN_MS)/1000
    
    # Find the maximum voltage in the stimulus trace (corresponds to a trigger)
    max_voltage = np.round(np.max(stimulus[:, 1]))

    # Round the voltage values for comparison to max voltage
    rounded_voltages = np.round(stimulus[:, 1])

    # Identify where the rounded voltage is equal to the rounded max voltage
    trigger_indices = np.where(rounded_voltages == max_voltage)[0]

    # Initialize an empty list for onset times
    onset_times = []

    # Loop over the identified trigger indices
    for index in trigger_indices:
        time = stimulus[index, 0]  # Get the corresponding time
        trigger_time_in_sec = time / 1000 + TRIGGER_DELAY_IN_MS / 1000  # Convert to seconds

        # If this is the first trigger, just append it
        if len(onset_times) == 0:
            onset_times.append(trigger_time_in_sec)
        else:
            # Since we know the triggers are 1.5s apart, we can safely append all valid triggers
            # Check if the current trigger is significantly different than the last (to avoid duplicates)
            if trigger_time_in_sec - onset_times[-1] >= trial_length:
                onset_times.append(trigger_time_in_sec)

    # Convert onset times to frames
    onset_frames_at_recording_fr = np.multiply(onset_times, RECORDING_FRAMERATE)

    return onset_frames_at_recording_fr  # Return the onset frames



def epoch_trace(fl, onset_frames,EPOCH_START_IN_MS,EPOCH_END_IN_MS,RECORDING_FRAMERATE):

    """
    Divide the fluorescence traces into trials (from ROI x frames array to ROI x trials x frames array)
    @param fl: the fluorescence traces given by Suite2P (nCells x nFrames)
    @param onset_frames: a vector of frames (at the fluo recording framerate) where the stims began
    @return epoched_traces: a nCells x nTrials x nFrames array where epoched_traces[0,0,:] would represent the fluo of the first trial of the first ROI
    """

    # Calculate the trial length in frames
    trial_length_in_frames = int((EPOCH_END_IN_MS - EPOCH_START_IN_MS) / 1000 * RECORDING_FRAMERATE)

    # Calculate start and end frames for each onset
    trial_start_frames = np.round(onset_frames + (EPOCH_START_IN_MS / 1000 * RECORDING_FRAMERATE)).astype(int)
    trial_end_frames = trial_start_frames + trial_length_in_frames

    # Ensure all trials fit within the bounds of the recording
    valid_indices = (trial_start_frames >= 0) & (trial_end_frames <= fl.shape[1])
    trial_start_frames = trial_start_frames[valid_indices]
    trial_end_frames = trial_end_frames[valid_indices]

    # Preallocate the array
    epoched_traces = np.zeros((fl.shape[0], len(trial_start_frames), trial_length_in_frames))

    # Use broadcasting to gather traces for all ROIs at once
    for trial_idx, (start, end) in enumerate(zip(trial_start_frames, trial_end_frames)):
        epoched_traces[:, trial_idx, :] = fl[:, start:end]

    return epoched_traces


def format_trials(traces,conditions):

    """
    Format the trials for a single ROI into a dictionary structure to store in the larger dictionary
    @param traces: nTrials x nFrames array of the fluorescence for each trial for a SINGLE ROI
    @param conditions: nTrial x 4 array detailing the frequency/intensity of each trial (see format_trials)
    @return freq_dict: a dictionary formatted according to the internal comments (each unique frequency one key, intensities as sub dictionaries, repetitions as sub-subdictionaries)
    """

    # traces should be an nTrial x nFrame array of the dF/F over each trial
    # stim should be an nTrial x 4 array (info on this structure in the README.md)

    # this will return a dictionary that will be contained within this cell's key in the recording dictionary

    # format the dictionary so we get this structure:
    # freq_f{
    #   intensity_i{
    #       repetition{ 
    #           [x,x,x,x,...] }}}

    # use the frequencies we played as the keys of our dictionary (outermost dictionary)
    freq_dict = dict.fromkeys(np.unique(conditions[:,0]))


    # nest each intensity the frequencies were presented at inside our freq dictionary
    for freq in freq_dict:
        freq_dict[freq] = dict.fromkeys(np.unique(conditions[:,1]))

    # make empty dictionaries so we can index properly later
    for freq in freq_dict:
        for intensity in freq_dict[freq]:
            freq_dict[freq][intensity] = {}

    # make a temporary map so we can keep track of how many repetitions of this trial we've seen
    # just going to add together the frequency and intensity to index it
    # biggest element we'll need is max(frequency) + max(intensity)
    max_element = max(conditions[:,0]) + max(conditions[:,1]) + 10
    temp_map = [0] * max_element

    # for each trial
    for trial in range(len(conditions)):

        # trial's frequency
        f = conditions[trial,0]

        # trial's intensity
        i = conditions[trial,1]

        # access the map to see how many repetitions of the frequency and intensity we've already seen
        # this way we don't overwrite a trial with the same stimulus type
        num_rep = temp_map[f+i]+1
        temp_map[f+i] += 1

        # using the frequency and intensity to index our dictionary to store our trace
        freq_dict[f][i][num_rep] = traces[trial,:]

    return freq_dict


def format_all_cells(epoched_traces,conditions,iscell_logical,epoched_deconvolved_traces):

    """
    Convert the epoched traces from an array to a dictionary
    @param epoched_traces: nCells x nTrials x nFrames array as returned from epoch_trace
    @param conditions: nTrials x 4 array where each row contains the frequency, intensity, trial delay, and stim length for a single trial 
    @param iscell_logigical: nCells x 1 vector with 1 or 0 value to designate whether the ROI is a cell (1) or not (0)
    @return dict_of_cells: dictionary where each cell is a key containing a subdictionary with the trials
    """

    # find the label for each ROI by finding this indices where iscell_logical is 1
    ROI_indices = (iscell_logical[:,0] == 1).nonzero()
    ROI_indices = ROI_indices[0] # extracting the first part of the tuple
    cell_IDs = ROI_indices + 1 # add 1 so we don't have zero indexing

    # make a dictionary from this list
    dict_of_cells = dict.fromkeys(cell_IDs)

    # for each cell
    # format the dictionary so we get this structure:
    # cell_n{ 
    #    'traces'{
    #           freq_f{
    #               intensity_i{
    #                     repetition{
    #                           [x,x,x,x,...] }}}}}

    for cell_idx in range(len(cell_IDs)):
        dict_of_cells[cell_IDs[cell_idx]] = {'traces': format_trials(epoched_traces[cell_idx,:,:],conditions),'deconvolved_traces': format_trials(epoched_deconvolved_traces[cell_idx,:,:],conditions)}
    
    return dict_of_cells


def main():

    # Load required files:

    stimulus = np.genfromtxt(BASE_PATH + CSV_PATH,delimiter=',',skip_header=True) # voltage values of the trigger software over the recording
    conditions_mat = scio.loadmat(BASE_PATH + CONDITIONS_PATH) # conditition type of each trial in chronological order (row 1 = trial 1)
    
    conditions = conditions_mat["stim_data"]
    fluorescence_trace = np.load(BASE_PATH + "F.npy",allow_pickle=True) # uncorrected traces of dF/F for each ROI
    deconvolved_trace = np.load(BASE_PATH + "spks.npy", allow_pickle=True ) # Deconvolved spike train for each ROI
    neuropil_trace = np.load(BASE_PATH + "Fneu.npy",allow_pickle=True) # estimation of background fluorescence for each ROI
    iscell_logical = np.load(BASE_PATH + "iscell.npy",allow_pickle=True) # Suite2P's estimation of whether each ROI is a cell or not

    stimulus_onset_frames = get_stimulus_onset_frames(stimulus)

    print(len(conditions)," stimuli were presented")
    print('Frequencies presented: {}'.format(np.unique(conditions[:,0])))
    print('Intensities presented: {}'.format(np.unique(conditions[:,1])))

    # Subtract background neuropil fluorescence from traces.
    corrected_fluo = fluorescence_trace - NEUROPIL_CORRECTION*neuropil_trace

    # Get the indices of ROIs that are actually cells only once
    cell_indices = np.where(iscell_logical[:, 0] == 1)[0]

    # Get fluorescence traces for the ROIs that are actually cells
    fluo_in_cells = corrected_fluo[cell_indices, :]

    # Get deconvolved traces for the ROIs that are actually cells
    deconvolved_in_cells = deconvolved_trace[cell_indices, :]

    # Epoch the traces using the obtained indices
    epoched_traces = epoch_trace(fluo_in_cells, stimulus_onset_frames)

    # epoch the deconvolved traces so we just get activity during trials
    epoched_deconvolved_traces = epoch_trace(deconvolved_in_cells,stimulus_onset_frames)

    np.save(BASE_PATH+"raw_corrected_traces.npy",fluo_in_cells) # save the trace for each cell ROI 
    np.save(BASE_PATH + "raw_deconvolved_traces.npy",deconvolved_in_cells) # save the deconvolved trace for each cell ROI
    np.save(BASE_PATH+"epoched_traces.npy",epoched_traces) # save the trace for trial before it's formatted into a dictionary
    np.save(BASE_PATH+"epoched_deconvolved_traces.npy",epoched_deconvolved_traces) # save the deconvolved trace for trial before it's formatted into a dictionary
    np.save(BASE_PATH+"onsets.npy",stimulus_onset_frames) # save the list of trigger frames (trial onsets)

    dictionary_of_cells = format_all_cells(epoched_traces,conditions,iscell_logical,epoched_deconvolved_traces)

    # collect some information about the stim to access later on if we want
    recording_info = dict()
    recording_info['frequencies'] = np.unique(conditions[:,0])
    recording_info['intensities'] = np.unique(conditions[:,1])
    recording_info['nRepeats'] = np.count_nonzero(np.logical_and(conditions[:,0]==recording_info['frequencies'][0],conditions[:,1] == recording_info['intensities'][0]))
    recording_info['nTrials'] = len(conditions)

    # Include all config information in the same dictionary
    recording_info.update(config.__dict__)

    # save to the provided output path
    with open(BASE_PATH+OUTPUT_PATH,'wb') as f:
        pickle.dump(dictionary_of_cells,f)

    # save the recording information 
    with open(BASE_PATH+"recording_info.pkl",'wb') as f:
        pickle.dump(recording_info,f)

if __name__=='__main__':
    main()

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Runtime: {elapsed_time:.2f} seconds")