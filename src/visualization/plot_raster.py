import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


STIMULUS_FRAMERATE = 100
TRIGGER_DELAY_IN_MS = 50 # delay between TDT sending a trigger and the stimulus actually happening
RECORDING_FRAMERATE = 10
SPIKE_THRESHOLD = 100
epoch_start_in_ms = -500 # in ms
epoch_end_in_ms = 6000 # in ms
stim_fl_error_allowed = 10 # time in seconds to allow as the difference in length between the stim file and fluorescence trace

BASE_PATH = "D:/vid127_pseudorandom_stim/"
epoched_traces_file = "epoched_traces.npy"

def get_spike_train(cell_trace):
    # we're getting an nTrials x nFrames array
    # but let's just make one continuous trace
    trace = cell_trace
    trace[trace<SPIKE_THRESHOLD] = 0
    trace[trace>SPIKE_THRESHOLD] = 1
    return trace

def get_raster_matrix(epoched_traces):
    # we will go cell by cell (row by row)
    # nCells x nTrials x nFrames array

    raster_matrix = np.zeros((len(epoched_traces),len(epoched_traces[1,:])))    
    # and count each frame above the threshold as 1
    for cell_idx in range(len(epoched_traces)):
        raster_matrix[cell_idx,:] = get_spike_train(epoched_traces[cell_idx,:])

    return raster_matrix

def plot_raster(raster_matrix,onsets):
    counter = 1
    for row in raster_matrix:
        raster = np.where(row==1)[0]
        plt.scatter(raster, np.ones(len(raster))*counter,marker="s",color="k",linewidths=0.75,alpha=0.7,s=2)
        counter+=1

    plt.xlabel("Frames (10 Hz)")
    plt.ylabel("Cell #")
    plt.title("Raster plot (spike threshold = 100 dF/F)")
    plt.suptitle("Pseudorandom stim")
    plt.vlines(onsets,0,counter,linewidth=2)

    plt.show()

def main():
    epoched_traces = np.load(BASE_PATH+epoched_traces_file,allow_pickle=True)
    onsets = np.load(BASE_PATH+"onsets.npy",allow_pickle=True)
    print(epoched_traces.shape)
    # print(epoched_traces)
    # print()

    raster_matrix = get_raster_matrix(epoched_traces)

    plot_raster(raster_matrix,onsets)

if __name__=="__main__":
    main()