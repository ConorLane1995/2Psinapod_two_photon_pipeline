import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from cmath import sqrt
import pandas as pd


def main():
    with open('/media/vtarka/USB DISK/Lab/2P/Vid_209/cells_ns.pkl', 'rb') as f:
            cell_dictionary = pickle.load(f)


    labels = pd.read_csv("/home/bic/vtarka/2Psinapod/Vid_209_activity.csv")

    print(labels['cell'][:5])  

    mismatch_counter = 0
    for cell in cell_dictionary:
        
        threshold_result = cell_dictionary[cell]['active']

        manual_result = labels.loc[labels['cell'] == cell, 'active'].iloc[0]

        if manual_result != 1:
            manual_result = False
        else:
            manual_result = True

        if threshold_result != manual_result:
            mismatch_counter += 1
            print(cell)

    print(mismatch_counter)

if __name__=="__main__":
    main()