from tkinter import N
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

BASE_PATH = "D:/vid127_pseudorandom_stim/"
traces_file = "cell_traces_with_stims.pkl"
output_file = "traces_with_activity_boolean.pkl"
EPOCH_START = -500
EPOCH_END = 2000
fs = 10

def check_cell(cell_trace,n_baseline_frames,STD_threshold):
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

def check_all_cells(traces,n_baseline_frames,STD_threshold):
    
    for cell in traces:
        if (check_cell(traces[cell],n_baseline_frames,STD_threshold)):
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


    traces_with_active_boolean = check_all_cells(traces,n_baseline_frames,STD_threshold)

    # find the number of active cells
    counter = 0
    for cell in traces_with_active_boolean:
        if traces_with_active_boolean[cell]['active'] == True:
            counter += 1

    print("Number of active cells: ")
    print(counter)

    # with open(BASE_PATH+output_file,'wb') as f:
    #     pickle.dump(traces_with_active_boolean,f)

if __name__=='__main__':
    main()