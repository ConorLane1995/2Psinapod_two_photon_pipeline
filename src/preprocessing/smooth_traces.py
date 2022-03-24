import numpy as np
import pickle
from scipy.stats import zscore

BASE_PATH = "D:/vid_148/"
traces_file = "cells.pkl"
output_file = "cells_smoothed.pkl"

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smooth_individual_cell(cell_traces):
    for freq in cell_traces:
        for intensity in cell_traces[freq]:
            for trial in cell_traces[freq][intensity]:
                cell_traces[freq][intensity][trial] = moving_average(cell_traces[freq][intensity][trial])

    return cell_traces

def main():
    with open(BASE_PATH + traces_file, 'rb') as f:
        cell_dictionary = pickle.load(f)

    for cell in cell_dictionary:
        cell_dictionary[cell]['traces'] = smooth_individual_cell(cell_dictionary[cell]['traces'])

    with open(BASE_PATH+output_file,'wb') as f:
        pickle.dump(cell_dictionary,f)

if __name__=='__main__':
    main()