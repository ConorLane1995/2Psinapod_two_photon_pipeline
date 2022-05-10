import numpy as np


def get_cell_x(cell):
    return cell["x"]


def get_cell_y(cell):
    return cell["y"]

def get_best_frequency_by_peak(cell):
    tuning_curve = cell['tuning_curve']
    max_response_idx = np.argmax(tuning_curve[:,1])
    return tuning_curve[max_response_idx,0]

def get_best_frequency_by_area(cell):
    tuning_curve = cell['tuning_curve_2']
    max_response_idx = np.argmax(tuning_curve[:,1])
    return tuning_curve[max_response_idx,0]

def get_active_cells(traces):

    # going to return a dictionary with only active cells, formatted exactly the same as traces

    d = dict.fromkeys(traces.keys())

    for cell in traces:
        if traces[cell]['active'] == True:
            d[cell] = traces[cell]
        else:
            d.pop(cell,None)

    return d

def get_entire_trace(cell):
    traces = cell["traces"]

    entire_trace = []
    for freq in traces:
        for intensity in traces[freq]:
            for repetition in traces[freq][intensity]:
                entire_trace.append(traces[freq][intensity][repetition])

    et = np.array(entire_trace)
    et = np.reshape(et,-1)
    return et

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
    # let's get a sum of all our traces to average later
    for freq in cell_trace:
        for intensity in cell_trace[freq]:
            for repetition in cell_trace[freq][intensity]:
                summed_traces[counter,:] = cell_trace[freq][intensity][repetition]
                counter += 1

    return np.average(summed_traces,axis=0)