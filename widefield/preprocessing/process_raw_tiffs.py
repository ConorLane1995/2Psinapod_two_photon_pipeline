
from skimage.io import imread
from skimage.measure import block_reduce
from skimage import restoration
import numpy as np
import os
import matplotlib.pyplot as plt

TRIGGER_DELAY_IN_MS  = 0
RECORDING_FRAMERATE = 10
EPOCH_END_IN_MS = 1000
EPOCH_START_IN_MS = 0

"""
Find the stimulus onsets from the trigger CSV and define as frames in the fluorescence recording
@param stimulus: 1D vector of the voltage trace of the stimulus triggers
@return onset_frames_at_recording_fr: a list of the frames in the fluo recording where the stim was presented
"""
def get_onset_frames(stimulus):
    # find the max voltage (this will be the value in the vector when the trigger was sent)
    max_voltage = max(stimulus, key=lambda x:x[1])
    max_voltage = max_voltage[1]

    onset_times = [] # empty list to append our onset frames into
    time_list_index = 0 # counter to keep track of our index in the onset_times list

    # for each frame in the stimulus file
    for stimulus_idx in range(len(stimulus)):
        (time,voltage) = stimulus[stimulus_idx] # unpack the voltage at that timepoint

        if voltage.round() == max_voltage.round(): # if the voltage was our trigger voltage
            if time_list_index == 0: # and if we're at the first index (so there's no previous index to compare with)
                trigger_time_in_sec = time/1000 + TRIGGER_DELAY_IN_MS/1000
                onset_times.append(trigger_time_in_sec) # add the time as an onset time in SECONDS
                time_list_index += 1

            # if we're not at index zero, we need to compare this voltage with the previous saved onset voltage
            # otherwise we save a bunch of voltages as separate triggers because they all match the max voltage
            # but we just want one timepoint per trigger
            elif time/1000 -  onset_times[time_list_index - 1] > 1: 
                trigger_time_in_sec = time/1000 + TRIGGER_DELAY_IN_MS/1000
                onset_times.append(trigger_time_in_sec) # want it in second not millisecond
                time_list_index += 1

    # get the onset times in terms of frames of our fluorescence trace
    onset_frames_at_recording_fr = np.multiply(onset_times,RECORDING_FRAMERATE) # s * f/s = f

    return onset_frames_at_recording_fr # a list of stimulus onsets in units of frames at recording framerate

"""
//TODO
"""
def baseline_adjust_pixel(px,onset_frames):
    # first we'll find how many seconds are in each trial (based on the chosen epoch start and end)
    trial_length_in_ms = EPOCH_END_IN_MS - EPOCH_START_IN_MS # this gives us length in ms
    trial_length_in_sec = trial_length_in_ms/1000 # now we have it in seconds

    # converting to frames (at the frame rate of the 2P recording)
    trial_length_in_frames = int(trial_length_in_sec * RECORDING_FRAMERATE) # s * f/s = f

    # and for each trial onset
    for trial_idx in range(len(onset_frames)-2):
            # get the trial starting frame and ending frame
            trial_starting_frame = np.round(onset_frames[trial_idx]) + (EPOCH_START_IN_MS/1000*RECORDING_FRAMERATE)
            trial_ending_frame = np.round(onset_frames[trial_idx]) + (EPOCH_END_IN_MS/1000*RECORDING_FRAMERATE)

            # grab this range of frames from the fl trace and store it in the epoched matrix
            trace = px[int(trial_starting_frame):int(trial_ending_frame)]

            # now grab the baseline period
            baseline_starting_frame = np.round(onset_frames[trial_idx]) - 300/1000*RECORDING_FRAMERATE
            baseline_ending_frame = np.round(onset_frames[trial_idx])

            baseline_trace = px[int(baseline_starting_frame):int(baseline_ending_frame)]
            baseline_avg = np.average(baseline_trace)

            adj_trace = np.subtract(trace,baseline_avg)
            px[int(trial_starting_frame):int(trial_ending_frame)]=adj_trace

    return px


def main():
    
    folder = "/media/vtarka/USB DISK/Widefield_Test/"
    images = [img for img in os.listdir(folder)]

    trigger_csv = np.genfromtxt("/media/vtarka/USB DISK/triggers.csv",delimiter=',',skip_header=True) # voltage values of the trigger software over the recording
    trigger_csv = trigger_csv[:,1:]
    # for each image in the image folder:
    # downsample image
    # concatenate the downsampled image to the 3D array
    video = []
    for img in images:
        im = imread(folder+img)
        downsamp_img = block_reduce(im,block_size=(8,8),func=np.mean)
        video.append(downsamp_img)

    video = np.array(video) # transform into a 3D numpy array

    # find the trigger frames
    onset_frames = get_onset_frames(trigger_csv)

    # for every pixel, epoch the fluorescence into trials
    baseline_adjusted_video = np.empty(shape=video.shape)

    for i in range(len(video[0])):
        for j in range(len(video[0][0])):
            baseline_adjusted_video[:,i,j] = baseline_adjust_pixel(video[:,i,j],onset_frames)

    plt.imshow(video[0,:,:])
    plt.show()

    psf = np.ones((5,5))
    for frame in range(len(video)):
        video[frame,:,:] = restoration.richardson_lucy(baseline_adjusted_video[frame,:,:],psf,200)

    plt.imshow(video[0,:,:])
    plt.show()

if __name__=="__main__":
    main()