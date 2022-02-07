import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

BASE_PATH = "D:/vid127_pseudorandom_stim/"
# traces_file = "cell_traces_with_stims2.pkl"
traces_file = "traces_with_activity_boolean.pkl"

def get_avg_trace(cell):

    # cell is going to be structured like this:
    # cell { freq { intensity { repetition: trace }}}

    # let's return a dictionary structured like this:
    # freq : avg_trace

    return_d = dict.fromkeys(cell.keys())

    # sum_trace = 0
    # num_rep = 0
    for freq in cell:

        for intensity in cell[freq]:
            sum_trace = 0
            num_rep = 0

            for rep in cell[freq][intensity]:
                sum_trace += cell[freq][intensity][rep]
                num_rep += 1

        avg_trace = sum_trace / num_rep
        return_d[freq] = avg_trace

    return return_d

def plot_trials(epoched_traces):
    # epoched traces is a dictionary structure as:
    # {cell : freq : intensity: repetition: trace}

    # get a random sampling of 20 cells
    # sample_idx = random.sample(range(len(epoched_traces)),20)
    all_active_cells = list(epoched_traces)
    cell_sample = random.sample(all_active_cells,20)

    # nCells = 20 #len(epoched_traces)
    # nFreqs = len(epoched_traces[1])

    fig, axs = plt.subplots(nrows=10, ncols=2)

    # get our averaged traces
    # make a dictionary with the cells as keys
    # plot_d = dict.fromkeys(range(1,11))

    for cell in range(10): 

        this_cell_d = get_avg_trace(epoched_traces[cell_sample[cell]])

        for trace in this_cell_d:
            # print(trace)
            axs[cell,0].plot(this_cell_d[trace]) 

        if cell != 9:
            axs[cell,0].get_xaxis().set_visible(False)
            axs[cell,0].get_yaxis().set_visible(False)
        
        axs[cell,0].vlines(5,0,50)
        axs[cell,0].set_ylim([0,50])
        axs[cell,0].set_xlim([0,32])

    for cell in range(10):
        this_cell_d = get_avg_trace(epoched_traces[cell_sample[cell+10]])

        for trace in this_cell_d:
            axs[cell,1].plot(this_cell_d[trace])

        if cell != 9:
            axs[cell,1].get_xaxis().set_visible(False)
            axs[cell,1].get_yaxis().set_visible(False)

        axs[cell,1].vlines(5,0,50)
        axs[cell,1].set_ylim([0,50])
        axs[cell,1].set_xlim([0,32])


    # for each cell, plot the average trace

    # for cell in plot_d:
    #     axs[cell-1].plot(plot_d[cell])

    plt.show()

def get_active_cells(traces):
    # going to return a dictionary with only active cells, formatted exactly the same as traces

    d = dict.fromkeys(traces.keys())

    for cell in traces:
        if traces[cell]['active'] == True:
            d[cell] = traces[cell]['traces']
        else:
            d.pop(cell,None)

    return d

def main():

    with open(BASE_PATH + traces_file, 'rb') as f:
        traces = pickle.load(f)
    
    # trace = get_avg_trace(traces[1])
    # print(trace)
    active_cells = get_active_cells(traces)
    print(len(active_cells))

    plot_trials(active_cells)

if __name__=="__main__":
    main()