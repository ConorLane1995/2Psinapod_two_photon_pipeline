"""
Script to take the files produced from the Suite2P preprocessing software and epoch the recording into trials.
INPUT: stim triggers in csv, Suite2P files (F.npy, Fneu.npy, iscell.npy)
OUTPUT: epoched_F.npy formatted as nCells x nTrials x nFrames array
AUTHOR: Veronica Tarka, January 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import compress
import numpy as np
import random
import pickle

## add another column to the output to store the ROI_ID of the cell
### SET VARIABLES FOR NOW - EVENTUALLY TAKE IT FROM THE CONFIG FILE OR A CSV WITH ALL RECORDINGS ###
STIMULUS_FRAMERATE = 100
TRIGGER_DELAY_IN_MS = 50 # delay between TDT sending a trigger and the stimulus actually happening
RECORDING_FRAMERATE = 10
epoch_start_in_ms = -500 # in ms
epoch_end_in_ms = 3000 # in ms
stim_fl_error_allowed = 10 # time in seconds to allow as the difference in length between the stim file and fluorescence trace

BASE_PATH = "D:/vid127_pseudorandom_stim/"
csv_path = "TSeries-01202022-1406-127_Cycle00001_VoltageRecording_001.csv"
output_path = "dictionary_of_cells_2.pkl"


def are_valid_files(stimulus,fluorescence):
    # make sure our stimulus file corresponds with our fluorescence trace
    # by comparing the length of stimulus file in seconds
    # against the length of the recording itself in seconds
    
    # get the total length of the stimulus and the recording in seconds
    stimulus_length_in_sec = len(stimulus)/STIMULUS_FRAMERATE # f * s/f = s
    fluorescence_length_in_sec = len(fluorescence[0])/RECORDING_FRAMERATE

    # if one is short or longer by more than the allowed error, return False so we can respond accordingly
    if stimulus_length_in_sec < fluorescence_length_in_sec -stim_fl_error_allowed \
         or stimulus_length_in_sec > fluorescence_length_in_sec + stim_fl_error_allowed:
        return False
    else:
        return True

def get_onset_frames(stimulus):
    # find the times when the stimulus occured, and convert it to frames at the recording's frame rate

    # get our max voltage value so we know what the trigger looks like
    max_voltage = max(stimulus, key=lambda x:x[1])
    max_voltage = max_voltage[1]

    onset_times = [] # empty list to append our onset frames into
    time_list_index = 0 # counter to keep track of our index in the onset_times list

    # for each frame in the stimulus file
    for stimulus_idx in range(len(stimulus)):
        (time,voltage) = stimulus[stimulus_idx] # unpack the voltage at that timepoint

        if voltage.round() == max_voltage.round(): # if the voltage was our trigger voltage
            if time_list_index == 0: # and if we're at the first index (so there's no previous index to compare with)
                trigger_time_in_sec = time/1000 + TRIGGER_DELAY_IN_MS/1000
                onset_times.append(trigger_time_in_sec) # add the time as an onset time in SECONDS
                time_list_index += 1

            # if we're not at index zero, we need to compare this voltage with the previous saved onset voltage
            # otherwise we save a bunch of voltages as separate triggers because they all match the max voltage
            # but we just want one timepoint per trigger
            elif time/1000 -  onset_times[time_list_index - 1] > 1: 
                trigger_time_in_sec = time/1000 + TRIGGER_DELAY_IN_MS/1000
                onset_times.append(trigger_time_in_sec) # want it in second not millisecond
                time_list_index += 1

    # get the onset times in terms of frames of our fluorescence trace
    onset_frames_at_recording_fr = np.multiply(onset_times,RECORDING_FRAMERATE) # s * f/s = f

    return onset_frames_at_recording_fr

def epoch_trace(fl,onset_frames):
    # fl is nCells x nFrames
    # frames is nOnsetTimes x 1

    # we will return an nCells x nTrials x nFrames array

    # first we'll find the number of frames that should be in each trial
    trial_length_in_ms = epoch_end_in_ms - epoch_start_in_ms # this gives us length in ms
    trial_length_in_sec = trial_length_in_ms/1000 # now we have it in seconds

    # converting to frames (at the frame rate of the 2P recording)
    trial_length_in_frames = int(trial_length_in_sec * RECORDING_FRAMERATE) # s * f/s = f

    # now we intitialize an array to store what we'll ultimately return
    epoched_traces = np.zeros((len(fl),len(onset_frames),trial_length_in_frames))

    # start filling up this empty matrix
    # loop through each ROI
    for roi_idx in range(len(fl)):

        # and for each trial onset
        for trial_idx in range(len(onset_frames)):

            # get the trial starting frame and ending frame
            trial_starting_frame = int(onset_frames[trial_idx] + (epoch_start_in_ms/1000*RECORDING_FRAMERATE))
            trial_ending_frame = int(onset_frames[trial_idx] + (epoch_end_in_ms/1000*RECORDING_FRAMERATE))

            # grab this range of frames from the fl trace and store it in the epoched matrix
            epoched_traces[roi_idx,trial_idx,:] = fl[roi_idx,trial_starting_frame:trial_ending_frame]

    return epoched_traces

def format_trials(traces,stim):

    # traces should be an nTrial x nFrame array
    # stim should be an nTrial x 4 array (info on this structure in the README.md)

    # need this to return a dictionary that will be contained within this cell key in the big dictionary

    # format the dictionary so we get this structure:
    # cell_ID{
    #     traces{ 
    #        freq_f{
    #           intensity_i{
    #                repetition{ 
    #                   trace = [x,x,x,x,...]
    #                           }
    #                   }
    #            }
    #       }
    # }

    # use the frequencies we played as the keys of our dictionary (outermost dictionary)
    freq_dict = dict.fromkeys(np.unique(stim[:,0]))

    # nest our intensities inside our freq dictionary
    for freq in freq_dict:
        freq_dict[freq] = dict.fromkeys(np.unique(stim[:,1]))

    # make empty dictionaries so we can index properly later
    for freq in freq_dict:
        # print(type(freq))
        for intensity in freq_dict[freq]:
            freq_dict[freq][intensity] = {}

    # make a really shitty temporary map so we can keep track of how many repetitions of this trial we've seen
    # just going to add together the frequency and intensity to index it
    # biggest element we'll need is max(frequency) + max(intensity)
    max_element = max(stim[:,0]) + max(stim[:,1]) + 10
    temp_map = [0] * max_element

    for trial in range(len(stim)):

        # trial's frequency
        f = stim[trial,0]

        # trial's intensity
        i = stim[trial,1]

        # access the map to see how many repetitions of the frequency and intensity we've already seen
        num_rep = temp_map[f+i]+1
        temp_map[f+i] += 1

        # using the frequency and intensity to index our dictionary to store our trace
        freq_dict[f][i][num_rep] = traces[trial,:]

    return freq_dict

def format_all_cells(epoched_traces,stimulus,iscell_logical):

    # make a dictionary where each cell is one key
    # enumerate all the ROI IDs
    ROI_IDs = range(1,len(iscell_logical)+1)
    # grab all the ROIs that are cells
    cell_IDs = list(compress(ROI_IDs, iscell_logical[:,0]==1))
    dict_of_cells = dict.fromkeys(cell_IDs)

    # for each cell
    # format the dictionary so we get this structure:
    # cell_n{ 
    #     freq_f{
    #           intensity_i{
    #                   trace = [x,x,x,x,...]
    #                       }
    #            }
    # }

    for cell_idx in range(len(cell_IDs)):
        dict_of_cells[cell_IDs[cell_idx]] = {'traces': format_trials(epoched_traces[cell_idx-1,:,:],stimulus)}
    
    return dict_of_cells

def plot_trace(fl,onsets):
    for i in range(len(fl)):
        plt.plot(fl[i,:]) #,label=i)

    # plt.legend(loc="upper left")
    plt.vlines(onsets,0,1000,linewidth=2)
    plt.show()

def plot_trials(epoched_traces,n_trial_samples,n_cell_samples):
    # epoched traces is an nCell x nTrial x nFrame matrix

    fig, axs = plt.subplots(nrows=n_trial_samples, ncols=1)

    # get a random sampling of 10 cells
    cell_sample = random.sample(range(len(epoched_traces)),n_cell_samples)

    # get a random sampling of trials
    trial_sample = random.sample(range(len(epoched_traces[0])),n_trial_samples)

    for trial in range(n_trial_samples):

        # for each cell we've sampled
        for cell in range(n_cell_samples):

            axs[trial].plot(epoched_traces[cell_sample[cell],trial_sample[trial],:])


        if trial == 0:
            axs[trial].set_title("Random sampling of cells and trials from the current recording")
            
        if trial != n_trial_samples-1:
            axs[trial].get_xaxis().set_visible(False)
            axs[trial].get_yaxis().set_visible(False)
        else:
            axs[trial].set_xlabel("Time since stimulus onset (ms)")

        # add a line to show exactly where stimulus happened
        # epoching started 100 ms before the trigger
        # so have an extra 0.1s * RECORDING_FRAMERATE frames before the trigger
        axs[trial].vlines(epoch_start_in_ms/-1000*RECORDING_FRAMERATE,0,100)

        # set the limits on the axes
        axs[trial].set_ylim([0,100])
        # axs[trial].set_xlim([0,0])

        # set the xticks so we're seeing the time
        axs[trial].set_xticks(range(1,20,5),range(0,2000,500))
        axs[trial].set_yticks([0,100])
        
    plt.show()


def main():

    # load our files
    stimulus = np.genfromtxt(BASE_PATH + csv_path,delimiter=',',skip_header=True)
    # conditions = np.load(BASE_PATH+"Stim_Data_PseudoRandom_vid127.npy",allow_pickle=True)
    fluorescence_trace = np.load(BASE_PATH + "F.npy",allow_pickle=True)
    neuropil_trace = np.load(BASE_PATH + "Fneu.npy",allow_pickle=True)
    iscell_logical = np.load(BASE_PATH + "iscell.npy",allow_pickle=True)

    # make sure the stim file and flu traces are roughly the same length
    # if they aren't the same, we'll exit the code 
    if not are_valid_files(stimulus, fluorescence_trace):
        raise ValueError("The lengths of stimulus file and the fluorescence traces are not the same. Exiting.")

    # get an array of all the stimulus onset times 
    # converted to be frames at the recording frame rate
    stimulus_onset_frames = get_onset_frames(stimulus)
    stimulus_onset_frames = stimulus_onset_frames[:-1] # remove the last element

    # account for the neuropil (background fluorescence)
    corrected_fluo = fluorescence_trace - 0.7*neuropil_trace

    # get fluorescence traces for the ROIs that are actually cells
    fluo_in_cells = corrected_fluo[np.where(iscell_logical[:,0]==1)[0],:]

    active_cells = range(21,41)

    print(stimulus_onset_frames)

    # plot_trace(fluo_in_cells[active_cells],stimulus_onset_frames)

    # epoch the traces so we just get the fluorescence during trials
    epoched_traces = epoch_trace(fluo_in_cells,stimulus_onset_frames)
    np.save(BASE_PATH+"epoched_traces.npy",fluo_in_cells)
    np.save(BASE_PATH+"onsets.npy",stimulus_onset_frames)

    # dictionary_of_cells = format_all_cells(epoched_traces,conditions,iscell_logical)

    # plot_trials(epoched_traces,8,15)

    # print(dictionary_of_cells)

    # save to the provided output path
    # with open(BASE_PATH+output_path,'wb') as f:
    #     pickle.dump(dictionary_of_cells,f)

if __name__=='__main__':
    main()