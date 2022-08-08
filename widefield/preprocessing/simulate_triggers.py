"""
Generate a CSV file with the same format as the usual trigger file.
Column 1: time stamps
Column 2: voltage
"""

import pandas as pd
import numpy as np

def main():
    # we want 7569 total entries in a column
    # Column 1 is going to contain the time stamps in ms
    # 10 frames per second, so start at 0 and increment in 100 until 756900
    df = pd.DataFrame()
    df['time'] = range(0,756900,100)

    # Column 2 is going to contain the "voltage" so 
    # We'll pass a trigger the first time at t=1000ms and then every 5000ms until 756900
    voltage = np.zeros(shape=(7569,))
    triggers = range(10,7569,50)
    voltage[triggers]=5
    df['voltage'] = voltage

    df.to_csv("/media/vtarka/USB DISK/Widefield_Test/triggers.csv",header=False)

if __name__=="__main__":
    main()