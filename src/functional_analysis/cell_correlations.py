import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../")
from utils import get_active_cells, get_entire_trace

BASE_PATH = "D:/vid_140/"
cell_dictionary = "cells.pkl"
coeffs_file = "corrcoefs.pkl"

def format_for_corrcoef(cell_dict):
    # each column is a single observation (trace)
    # each row is a variable (cell)

    data = []
    for cell in cell_dict:
        data.append(get_entire_trace(cell_dict[cell]))

    data = np.array(data)
    return data

def main():
    with open(BASE_PATH+cell_dictionary,"rb") as f:
        cell_dict = pickle.load(f)

    # get only the active cells
    active_cell_dict = get_active_cells(cell_dict)

    active_cell_dict_formatted = format_for_corrcoef(active_cell_dict)
    # print(active_cell_dict_formatted)

    corr_matrix = np.corrcoef(active_cell_dict_formatted)

    with open(BASE_PATH+coeffs_file,"wb") as f:
        pickle.dump(corr_matrix,f)

if __name__=="__main__":
    main()