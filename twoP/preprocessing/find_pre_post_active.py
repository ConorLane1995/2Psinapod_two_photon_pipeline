"""
TODO doc
Script to find 
"""

import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells

def main():
    
    with open("/media/vtarka/USB DISK/Lab/2P/Vid_246/cells.pkl","rb") as f:
        r1_cells = pickle.load(f)

    with open("/media/vtarka/USB DISK/Lab/2P/Vid_247/cells.pkl","rb") as f:
        r2_cells = pickle.load(f)

    r1_acells = get_active_cells(r1_cells)
    r2_acells = get_active_cells(r2_cells)

    total_acells = list(r1_acells.keys()) + list(r2_acells.keys())

    unique_acells = np.unique(total_acells)

    np.save("/media/vtarka/USB DISK/Lab/2P/Vid_247/active_cells.npy",unique_acells)

    
if __name__=="__main__":
    main()