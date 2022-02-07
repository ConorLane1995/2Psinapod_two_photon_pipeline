"""
Script to find which of the cells that were recorded are actually active.
INPUT: Dictionary with the epoched traces
OUTPUT: Same dictionary but now with an added key "active" that holds a boolean
AUTHOR: Veronica Tarka, January 2022, veronica.tarka@mail.mcgill.ca
"""

from tkinter import N
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import zscore

BASE_PATH = "D:/vid127_pseudorandom_stim/"
traces_file = "cell_traces_with_stims.pkl"
output_file = "traces_with_activity_boolean.pkl"
EPOCH_START = -500
EPOCH_END = 2700
fs = 10

def check_cell_STD(cell_trace,n_baseline_frames,STD_threshold):
    # cell_trace is going to be all the trials of this one cell
    # {freq: intensity: repetition: trace = [x,x,x,x,...]}}}

    # we'll say a cell is active if it reaches beyond the peak threshold on >1 trial on >0 frequencies
    reached_threshold_counter = 0 # need this to be >1 to return true

    # go through each frequency
    for freq in cell_trace:

        # go through each intensity it was played at:
        for intensity in cell_trace[freq]:

            # go through ecah repetition
            for repetition in cell_trace[freq][intensity]:

                this_trial = cell_trace[freq][intensity][repetition]

                # get the baseline for this trial
                baseline = this_trial[0:n_baseline_frames-1]
                response = this_trial[n_baseline_frames:]

                # get our threshold
                peak_threshold = np.mean(baseline) + STD_threshold*np.std(baseline)

                # get the peak response
                peak_response = np.amax(response)

                # if our peak response was above our threshold, increase the counter
                if peak_response >= peak_threshold:
                    reached_threshold_counter += 1

    if reached_threshold_counter > 1:
        return True
    else:
        return False

def check_cell_zscore(cell_trace,zscore_threshold):
    # cell_trace is going to be all the trials of this one cell
    # {freq: intensity: repetition: trace = [x,x,x,x,...]}}}

    active = False

    # get our average trace
    avg_trace = get_avg_trace(cell_trace)

    # convert it to z scores
    avg_trace_zscores = zscore(avg_trace)

    # divide it into baseline and response
    onset = round(EPOCH_START/1000 * fs * -1) # how many extra frames we have at the beginning before our stim onset
    #baseline = avg_trace_zscores[:onset-1]
    response = avg_trace_zscores[onset:]

    # get the peak of the response
    peak_response = np.amax(response)

    if peak_response > zscore_threshold:
        active = True

    return active

# HELPER METHOD THAT SHOULD ULTIMATELY GO IN SRC!!
def get_avg_trace(cell_trace):
    # cell_trace is going to be all the trials of this one cell
    # {freq: intensity: repetition: trace = [x,x,x,x,...]}}}

    # first we need to find how much space to allocate
    n_samples = 0
    n_trials = 0
    for freq in cell_trace:
        for intensity in cell_trace[freq]:
            for repetition in cell_trace[freq][intensity]:
                if n_trials == 0:
                    n_samples = len(cell_trace[freq][intensity][repetition])
                n_trials += 1
              
    summed_traces = np.zeros(shape=(n_trials,n_samples))

    counter = 0
    # let's get a sum of all our traces
    for freq in cell_trace:
        for intensity in cell_trace[freq]:
            for repetition in cell_trace[freq][intensity]:
                summed_traces[counter,:] = cell_trace[freq][intensity][repetition]
                counter += 1

    return np.average(summed_traces,axis=0)
    

def check_all_cells(traces):
    
    for cell in traces:
        if (check_cell_zscore(traces[cell]['traces'],2)):
            traces[cell]['active'] = True
        else:
            traces[cell]['active'] = False

    return traces

def main():
    
    # import our epoched and formatted recordings
    # again, it's formatted like this: 
    # cell { freq { intensity { repetition: trace = [x,x,x,x,...] }}}
    with open(BASE_PATH + traces_file, 'rb') as f:
        traces = pickle.load(f)

    # define some key variables we'll pass into our functions
    late_peak = 1.5 - (EPOCH_START/1000) * fs # any peaks more than 1.5 seconds after onset will be considered late
    n_baseline_frames = round(EPOCH_START/1000 * fs) # these are the frames we'll use as the baseline
    STD_threshold = 3 # number of standard deviations from baseline


    traces_with_active_boolean = check_all_cells(traces)

    # find the number of active cells
    counter = 0
    for cell in traces_with_active_boolean:
        if traces_with_active_boolean[cell]['active'] == True:
            counter += 1
            print(cell)

    print("Number of active cells: ")
    print(counter)

    # print(len(get_avg_trace(traces[1])))
    # print(len(traces[1][1000][60][1]))
    with open(BASE_PATH+output_file,'wb') as f:
        pickle.dump(traces_with_active_boolean,f)

if __name__=='__main__':
    main()