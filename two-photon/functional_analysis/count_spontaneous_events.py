from cmath import sqrt
from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import sys
import json
import os
from scipy.stats import zscore

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
cell_dictionary_file = config['AnalysisFile']
cell_dictionary_file_out = cell_dictionary_file
EPOCH_START_IN_MS = config['EpochStart']
EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
FRAMERATE = config['RecordingFR']

ZSCORE_THRESHOLD = 2
WINDOW_SIZE = 5

def get_frame_zscore(baseline,frame):
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    frame_zscore = (frame - baseline_mean) / baseline_std

    return frame_zscore

def get_unique_events(frames):
    frames = frames[0]
    unique_events = []

    event_idx = 0
    while event_idx < len(frames):
        if event_idx == 0:
            unique_events.append(frames[event_idx])

        previous_event = frames[event_idx-1]
        this_event = frames[event_idx]

        if (this_event - previous_event > 10):
            unique_events.append(frames[event_idx])

        event_idx += 1

    return unique_events


def count_events(cell_trace):

    frames = []
    # frame = WINDOW_SIZE
    
    # while frame < len(cell_trace):
    #     # print(frame)
    #     # input("Press enter...")

    #     frame_zscore = get_frame_zscore(cell_trace[frame-WINDOW_SIZE:frame],cell_trace[frame])

    #     if frame_zscore>ZSCORE_THRESHOLD:
    #         # print("event")
    #         frames.append(frame)
    #         frame += 10
    #     else:
    #         frame += 1

    zscored_trace = zscore(cell_trace)
    frames = np.nonzero(zscored_trace > ZSCORE_THRESHOLD)

    unique_frames = get_unique_events(frames)

    return unique_frames

def main():
    # for each cell
    # look at the 10 frames preceding the current frame
    # zscore in relation to these frames
    # if the zscore is more than 3?? z-scores, it is an event
    # may need to implement a skip-forward after event identification..
    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    event_frames = []
    for cell in cell_dictionary:
        event_frames.append(count_events(cell_dictionary[cell]['traces']))

    # plt.hist(total_events)
    # plt.show()
    
    with open(BASE_PATH+'events.pkl','wb') as f:
        pickle.dump(event_frames,f)

if __name__=="__main__":
    main()