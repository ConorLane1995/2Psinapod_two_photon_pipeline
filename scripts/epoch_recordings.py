import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv

stim_fr = 1000
flu_fr = 10
epoch_start = -100
epoch_end = 2000

csv_path = "C:/Users/vmtar/2Psinapod/data/TSeries-12202021-1139-113_Cycle00001_VoltageRecording_001.csv"
f_path = "C:/Users/vmtar/2Psinapod/data/F.npy"
fneu_path = "C:/Users/vmtar/2Psinapod/data/Fneu.npy"
iscell_path = "C:/Users/vmtar/2Psinapod/data/iscell.npy"


def validate_lengths(stim,fl):
    
    stim_time = len(stim)/stim_fr
    fl_time = len(fl[0])/flu_fr

    if stim_time < fl_time-10 or stim_time > fl_time+10:
        return False
    else:
        return True


def get_onset_frames(stim):
    
    # # find the frames when stim == max(stim)
    # o_frames_idx = stim[:,1] == np.amax(stim[:,1])

    # print(stim[:,0])

    # # use our logical array to find the frames when the stimulus occured
    # o_frames = stim[o_frames_idx,0]
    # print(o_frames)

    max_value = np.amax(stim[:][1].round())
    testing = np.where(stim[:][1].round() == max_value)
    print(testing)

    o_frames = np.take(stim,testing,0)
    print(o_frames)

    # find the actual times when the stimulus occured so we can convert it to flu frames
    o_times = o_frames/stim_fr

    # get the flu frames when the onsets occured
    o_frames_flu_trace = o_times*stim_fr

    return o_frames_flu_trace




def main():

    # with open(csv_path,'r') as csvfile:
    #     stim = csv.reader(csvfile,skip_header=True)

    stim = np.genfromtxt(csv_path,delimiter=',',dtype=None,skip_header=True)
    fl = np.load(f_path)
    fneu = np.load(fneu_path)
    iscell = np.load(iscell_path)

    fl = fl[:,800:len(fl[0])]
    print(stim[1][1])

    # make sure the stim file and flu traces are roughly the same length
    # if they aren't the same, we'll exit the code 
    if not validate_lengths(stim, fl):
        raise ValueError("The lengths of stimulus file and the fluorescence traces are not the same. Exiting.")


    print(stim.shape)
    # get an array of all the stimulus onset times in terms of fluorescence frames
    onset_frames = get_onset_frames(stim)

    print(onset_frames)

    # fig, ax = plt.subplots()
    # ax.plot(stim[:,1])

    # plt.show()


if __name__=='__main__':
    main()