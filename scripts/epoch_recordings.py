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

    # get our max voltage value so we know what the trigger looks like
    max_value = max(stim, key=lambda x:x[1])
    max_value = max_value[1]

    onset_stim_frames = []
    # get idx where the trigger was sent (stim onset)
    for i in range(len(stim)):
        (frame,voltage) = stim[i]
        if voltage.round() == max_value.round():
            onset_stim_frames.append(frame)

    # eliminate our extra frames included because they were close to the max voltage value
    onset_stim_frames_good = []
    curr_frame = onset_stim_frames[0]
    onset_stim_frames_good.append(curr_frame)
    for i in onset_stim_frames:
        if i - curr_frame > 1000:
            curr_frame = i
            onset_stim_frames_good.append(i)

    # find the actual times when the stimulus occured so we can convert it to flu frames
    onset_stim_times = [x / stim_fr for x in onset_stim_frames_good]

    # get the flu frames when the onsets occured
    onset_flu_frames = [x*flu_fr for x in onset_stim_times]

    return onset_flu_frames




def main():

    # with open(csv_path,'r') as csvfile:
    #     stim = csv.reader(csvfile,skip_header=True)

    stim = np.genfromtxt(csv_path,delimiter=',',skip_header=True)
    fl = np.load(f_path)
    fneu = np.load(fneu_path)
    iscell = np.load(iscell_path)

    fl = fl[:,800:len(fl[0])]

    # make sure the stim file and flu traces are roughly the same length
    # if they aren't the same, we'll exit the code 
    if not validate_lengths(stim, fl):
        raise ValueError("The lengths of stimulus file and the fluorescence traces are not the same. Exiting.")

    # get an array of all the stimulus onset times in terms of fluorescence frames
    onset_frames = get_onset_frames(stim)

    print(onset_frames)

    # fig, ax = plt.subplots()
    # ax.plot(stim[:,1])

    # plt.show()


if __name__=='__main__':
    main()