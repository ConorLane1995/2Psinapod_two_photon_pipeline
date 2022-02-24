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