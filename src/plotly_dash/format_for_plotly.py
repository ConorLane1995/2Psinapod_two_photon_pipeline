import numpy as np
import pandas as pd
import pickle
import random

BASE_PATH = "D:/vid127_pseudorandom_stim/"
traces_file = "traces_with_activity_boolean_2.pkl"
avgs_file = "active_cell_averages.pkl"

def get_active_cells(traces):
    # going to return a dictionary with only active cells, formatted exactly the same as traces

    d = dict.fromkeys(traces.keys())

    for cell in traces:
        if traces[cell]['active'] == True:
            d[cell] = traces[cell]['traces']
        else:
            d.pop(cell,None)

    return d

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


def main():
    with open(BASE_PATH + traces_file, 'rb') as f:
        cells = pickle.load(f)

    active_cells = get_active_cells(cells)

    active_cell_avgs = {}
    for cell in active_cells:
        active_cell_avgs[cell] = get_avg_trace(active_cells[cell])

    with open(BASE_PATH + avgs_file, 'wb') as f:
        pickle.dump(active_cell_avgs,f)



if __name__ == "__main__":
    main()