import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv

stim_fr = 100
flu_fr = 10
epoch_start = -100
epoch_end = 2000

csv_path = "D:/15khz/TSeries-01042022-1431-120_Cycle00001_VoltageRecording_001.csv"
f_path = "D:/mismatched_recording/F.npy"
fneu_path = "D:/mismatched_recording/Fneu.npy"
iscell_path = "D:/mismatched_recording/iscell.npy"

parent_dir = "D:/15khz/"

def validate_lengths(stim,fl):
    
    stim_time = len(stim)/stim_fr
    fl_time = len(fl[0])/flu_fr

    # print(stim_time)
    # print(fl_time)
    if stim_time < fl_time-10 or stim_time > fl_time+10:
        return False
    else:
        return True


def get_onset_frames(stim):

    # get our max voltage value so we know what the trigger looks like
    max_value = max(stim, key=lambda x:x[1])
    max_value = max_value[1]

    onset_times = []
    time_list_index = 0
    for i in range(len(stim)):
        # print(i)
        # print(time_list_index)
        (time,voltage) = stim[i]
        if voltage.round() == max_value.round():
            if time_list_index == 0:
                onset_times.append(time/1000)
                time_list_index += 1
            elif time/1000 -  onset_times[time_list_index - 1] > 1: 
                onset_times.append(time/1000) #want it in second not millisecond
                time_list_index += 1


    # onset_stim_times = []
    # # get idx where the trigger was sent (stim onset)
    # for i in range(len(stim)):
    #     (time,voltage) = stim[i]
    #     if voltage.round() == max_value.round():
    #         onset_stim_times.append(time)

    # # eliminate our extra frames included because they were close to the max voltage value
    # onset_stim_times_good = []
    # curr_frame = onset_stim_times[0]
    # onset_stim_times_good.append(curr_frame)
    # for i in onset_stim_times:
    #     if i - curr_frame > 2*stim_fr: # triggers are definitely at least 2 seconds apart
    #         curr_frame = i
    #         onset_stim_times_good.append(i)

    # print(onset_times)

    # find the actual times when the stimulus occured so we can convert it to flu frames
    # onset_stim_times = [x / stim_fr for x in onset_stim_frames_good]

    # get the flu frames when the onsets occured
    onset_flu_frames = [x * flu_fr for x in onset_times]

    return onset_flu_frames


def plot_trace(fl,onsets):
    for i in range(len(fl)):
        plt.plot(fl[i,:])

    plt.vlines(onsets,0,200)
    plt.show()


def main():

    # with open(csv_path,'r') as csvfile:
    #     stim = csv.reader(csvfile,skip_header=True)

    stim = np.genfromtxt(csv_path,delimiter=',',skip_header=True)
    fl = np.load(parent_dir + "F.npy")
    fneu = np.load(parent_dir + "Fneu.npy")
    iscell = np.load(parent_dir + "iscell.npy")

    #fl = fl[:,800:len(fl[0])]

    # make sure the stim file and flu traces are roughly the same length
    # if they aren't the same, we'll exit the code 
    if not validate_lengths(stim, fl):
        raise ValueError("The lengths of stimulus file and the fluorescence traces are not the same. Exiting.")

    # get an array of all the stimulus onset times in terms of fluorescence frames
    onset_frames = get_onset_frames(stim)

    # get fluorescence traces for the ROIs that are actually cells
    fl_cells = fl[np.where(iscell[:,0]==1)[0],:]
    #np.take_along_axis(fl,np.where(iscell[:,0]==1),1)

    # print(np.take(fl,np.where(iscell[:,0]==1)).shape)
    # print(len(np.where(iscell[:,0]==1)[0]))

    plot_trace(fl_cells,onset_frames)

    # plt.plot(fl[1,:])
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(stim[:,1])

    # plt.show()


if __name__=='__main__':
    main()