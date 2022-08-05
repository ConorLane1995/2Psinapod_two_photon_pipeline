import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os


np.seterr(divide='ignore', invalid='ignore')

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'\..\..\config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
# cell_dictionary_file = config['AnalysisFile']
# EPOCH_START_IN_MS = config['EpochStart']
# EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
# FRAMERATE = config['RecordingFR']

EVENT_THRESHOLD = 200

def count_num_events(traces):
    return np.count_nonzero(traces > EVENT_THRESHOLD)

def get_avg_event_amplitude(traces):
    # get the events 
    events = traces[traces > EVENT_THRESHOLD]
    if np.size(events, axis=None) != 0:
        return np.average(events,axis=None)
    else:
        return 0


def main():
    # two things we want to look at :
    # absolute number of events in the first half vs the second half of the recording
    # average magnitude of these events in 1st vs 2nd half
    
    # epoched_traces = np.load(BASE_PATH+"epoched_traces.npy",allow_pickle=True)
    raw_traces = np.load(BASE_PATH+"raw_corrected_traces.npy",allow_pickle=True)
    # onsets = np.load(BASE_PATH+"onsets.npy",allow_pickle=True)

    print(raw_traces[:,:round(len(raw_traces[0,:])/2)].shape)

    freq_diffs = []
    ampl_diffs = []
    for row in raw_traces:

        n_events_1 = count_num_events(row[:round(len(row)/2)])
        n_events_2 = count_num_events(row[round(len(row)/2):])
        freq_diffs.append(n_events_1 - n_events_2)

        ampl_1 = get_avg_event_amplitude(row[:round(len(row)/2)])
        ampl_2 = get_avg_event_amplitude(row[round(len(row)/2):])
        ampl_diffs.append(ampl_1 - ampl_2)


    # count the number of events in the first half vs the second half
    num_events_first_half = count_num_events(raw_traces[:,:round(len(raw_traces[0,:])/2)])
    num_events_second_half = count_num_events(raw_traces[:,round(len(raw_traces[0,:])/2):])

    avg_event_amplitude_first_half = get_avg_event_amplitude(raw_traces[:,:round(len(raw_traces[0,:])/2)])
    avg_event_amplitude_second_half = get_avg_event_amplitude(raw_traces[:,round(len(raw_traces[0,:])/2):])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(freq_diffs,bins=[-100,0,100,200,300,400,500])
    ax1.set_xlabel("# of events in 1st half - 2nd half")
    ax1.set_ylabel("# of cells")
    ax2.hist(ampl_diffs,bins=[-150,-100,-50,0,50,100,150])
    ax2.set_xlabel("Average event amplitude in 1st half - 2nd half")

    plt.show()

if __name__=="__main__":
    main()