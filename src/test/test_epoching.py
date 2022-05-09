# pick three random trials
# find which trigger they align with
# make sure the difference between the trial it should be and the recorded trace is 0
import pickle
from scipy.stats import zscore
import scipy.io as scio
import numpy as np

BASE_PATH = "/Volumes/Office_USB/Vid_157/"
traces_file = "cells_2.pkl"
output_file = "cells_2.pkl"
conditions_path = "Stim_Data_157_Corrected.mat"

def main():
    # import cells file
    # import our epoched and formatted recordings
    # again, it's formatted like this: 
    # cell { traces { freq { intensity { repetition: trace = [x,x,x,x,...] }}}}
    with open(BASE_PATH + traces_file, 'rb') as f:
        cell_dictionary = pickle.load(f)


    # import conditions file
    conditions_mat = scio.loadmat(BASE_PATH + conditions_path)
    conditions = conditions_mat["stim_data"]

    to_try = []

    for cell in cell_dictionary:
        t = cell_dictionary[cell]['traces']
        for freq in t:
            for intensity in t[freq]:
                counter = 0
                for rep in t[freq][intensity]:
                    to_try.append(t[freq][intensity][rep])
                    counter+=1
                    if counter>2:
                        break

                break
            break
        break

    to_try = cell_dictionary[1]['traces'][2000][90][1]
    fluorescence_trace = np.load(BASE_PATH + "F.npy",allow_pickle=True)
    neuropil_trace = np.load(BASE_PATH + "Fneu.npy",allow_pickle=True)

    fluorescence_trace = fluorescence_trace - 0.7*neuropil_trace

    onsets = np.load(BASE_PATH+"onsets.npy",allow_pickle=True)

    # find which onsets correspond with the first three presentations of 2000 @ 40 dB
    onsets_of_interest = np.where((conditions[:,0]==2000) & (conditions[:,1]==90))
    print(onsets[onsets_of_interest[0][0]])

    t1_onset = int(onsets[onsets_of_interest[0][0]])
    trial_1_f = fluorescence_trace[0,(t1_onset-5):(t1_onset+25)]

    print(trial_1_f.shape)
    to_try = np.array(to_try)
    # print(to_try[0,:])
    trial1_diff = trial_1_f - to_try#[0,:]
    print(np.unique(trial1_diff))


    


if __name__=="__main__":
    main()