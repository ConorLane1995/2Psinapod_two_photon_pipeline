import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from plot_psth import get_active_cells

BASE_PATH = "D:/Vid_139/"
cell_dictionary_file = "cells.pkl"
EPOCH_START_IN_MS = -500 # time before trial onset included in the epoch
EPOCH_END_IN_MS = 2500 # time after trial onset included in the epoch
FRAMERATE = 10

def get_best_frequency(cell):
    tuning_curve = cell['tuning_curve']
    max_response_idx = np.argmax(tuning_curve[:,1])
    return tuning_curve[max_response_idx,0]

def plot_tuning_curves(active_cells):
        # epoched traces is a dictionary structure as:
    # {cell : freq : intensity: repetition: trace}

    # get a random sampling of 20 cells
    # sample_idx = random.sample(range(len(epoched_traces)),20)
    all_active_cells = list(active_cells.keys())
    cell_sample = random.sample(all_active_cells,57)

    # print(cell_sample)
    # print()

    # fig, axs = plt.subplots(nrows=10, ncols=1)


    # get our averaged traces
    # make a dictionary with the cells as keys
    # plot_d = dict.fromkeys(range(1,11))

    cell_idx = 0
    for cell in cell_sample: 
        # print(cell)

        this_cell_tuning_curve = active_cells[cell]['tuning_curve']

        # axs[cell_idx].plot(this_cell_tuning_curve[:,0],this_cell_tuning_curve[:,1]) 

        # if cell_idx != 9:
        #     axs[cell_idx].get_xaxis().set_visible(False)
        #     axs[cell_idx].get_yaxis().set_visible(False)
        
        # # axs[cell,0].vlines(5,0,50)
        # # axs[cell,0].set_ylim([0,50])
        # #axs[cell_idx].set_xlim([0,32])

        # cell_idx += 1

        plt.plot(this_cell_tuning_curve[:,0],this_cell_tuning_curve[:,1])

    # for cell_idx in range(10):
    #     this_cell_tuning_curve = all_active_cells[cell_sample[cell_idx+10]]['tuning_curves']

    #     axs[cell_idx,1].plot(this_cell_tuning_curve[:,0],this_cell_tuning_curve[:,1]) 

    #     if cell_idx != 9:
    #         axs[cell_idx,1].get_xaxis().set_visible(False)
    #         axs[cell_idx,1].get_yaxis().set_visible(False)
        
    #     # axs[cell,0].vlines(5,0,50)
    #     # axs[cell,0].set_ylim([0,50])
    #     axs[cell_idx,1].set_xlim([0,32])



    plt.show()

def plot_best_frequency_histogram(active_cells):
    
    bfs = []

    for cell in active_cells:
        bfs.append(get_best_frequency(active_cells[cell]))

    plt.hist(bfs)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Number of cells")
    plt.title("Best frequencies of responsive cells")
    plt.show()

def main():

    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)
    
    active_cell_dictionary = get_active_cells(cell_dictionary)
    # print(active_cell_dictionary[204]['tuning_curve'])
    print(len(active_cell_dictionary))

    # plot_tuning_curves(active_cell_dictionary)

    plot_best_frequency_histogram(active_cell_dictionary)

if __name__=="__main__":
    main()