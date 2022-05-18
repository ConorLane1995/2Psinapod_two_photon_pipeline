import numpy as np
import pickle

BASE_PATH = "D:\\Lab\\2P\\Vid_176\\"
cell_dictionary_file = "cells_std.pkl"

def main():
    stat = np.load(BASE_PATH+"stat.npy",allow_pickle=True)
    
    # load our dictionary
    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dict = pickle.load(f)

    # add the x and y coordinates to the cell dictionary
    for cell in cell_dict:
        this_cell_x = stat[cell-1]["med"][0]
        this_cell_y = stat[cell-1]["med"][1]

        cell_dict[cell]['x'] = this_cell_x
        cell_dict[cell]['y'] = this_cell_y
    
    # save it
    with open(BASE_PATH + cell_dictionary_file, 'wb') as f:
        pickle.dump(cell_dict,f)

if __name__=="__main__":
    main()