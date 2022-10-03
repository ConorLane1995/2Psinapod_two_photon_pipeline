import time
from datetime import timedelta
start_time = time.monotonic()


import numpy as np
from skimage.io import imread
from skimage.measure import block_reduce
import os
from matplotlib import pyplot as plt
import tifffile
import scipy.io as sio
import pickle

TRIGGER_DELAY_IN_MS = 0
RECORDING_FRAMERATE = 10.01
EPOCH_START_IN_MS = -500
EPOCH_END_IN_MS = 2000




#CONVERT THIS TO A FUNCTION:

# Take the mean of each pixel across all frames and store it in a 256x256 ndarray. 
#mean_pixels = np.empty(shape=[256,256])

#for i in range(len(video[0,:,0])):
        #for j in range(len(video[0,0,:])):

                #mean = np.mean(video[:,i,j])
                #mean_pixels[i,j] = mean

#For each frame, subtract the mean of that frame from the total recording, 
# then divide by the mean to get (F-Fo)/Fo

#Reshape array so can subtract mean value
#mean_pixels = mean_pixels[np.newaxis,...]
#baseline_subtracted = (np.subtract(video,mean_pixels))
#DeltaF_Fo = baseline_subtracted/mean_pixels




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

    #Remove first trigger at frame zero as this is simply to start the recording
    onset_frames_at_recording_fr = onset_frames_at_recording_fr[1:]

    return onset_frames_at_recording_fr

def baseline_adjust_pixel(px,onset_frames):
    # first we'll find how many seconds are in each trial (based on the chosen epoch start and end)
    trial_length_in_ms = EPOCH_END_IN_MS - EPOCH_START_IN_MS # this gives us length in ms
    trial_length_in_sec = trial_length_in_ms/1000 # now we have it in seconds

    # converting to frames (at the frame rate of the 2P recording)
    trial_length_in_frames = int(trial_length_in_sec * RECORDING_FRAMERATE)+1 # s * f/s = f

    # and for each trial onset
    for trial_idx in range(len(onset_frames)-2):
            # get the trial starting frame and ending frame
            trial_starting_frame = np.round(onset_frames[trial_idx]) + (EPOCH_START_IN_MS/1000*RECORDING_FRAMERATE)
            trial_ending_frame = np.round(onset_frames[trial_idx]) + (EPOCH_END_IN_MS/1000*RECORDING_FRAMERATE)

            # grab this range of frames from the fl trace and store it in the epoched matrix
            trace = px[int(trial_starting_frame)-5:int(trial_ending_frame)]

            # now grab the baseline period
            baseline_starting_frame = np.round(onset_frames[trial_idx]) - 500/1000*RECORDING_FRAMERATE
            baseline_ending_frame = np.round(onset_frames[trial_idx])

            baseline_trace = px[int(baseline_starting_frame):int(baseline_ending_frame)]
            baseline_avg = np.average(baseline_trace)

            adj_trace = np.subtract(trace,baseline_avg)
            px[int(trial_starting_frame):int(trial_ending_frame)]=adj_trace

    return px


def epoch_trials(video,onset_frames):

        # Get length of trial in seconds
        trial_length_in_ms = EPOCH_END_IN_MS - EPOCH_START_IN_MS # this gives us length in ms
        trial_length_in_sec = trial_length_in_ms/1000 # now we have it in second

        # Convert this to length in frames
        trial_length_in_frames = int(trial_length_in_sec * RECORDING_FRAMERATE) # s * f/s = f

        # Initialize an array to store the epoched traces
        # nTrials x nFrames x nPixels x nPixels

        epoched_pixels = np.zeros((len(onset_frames),(trial_length_in_frames)+1, len(video[0,:,0]), len(video[0,0,:])))

        #Start filling the empty matrix:
        # Loop through the onset frames

        for onset in range(len(onset_frames)-1):

                #Get the trial starting and ending frames
                trial_starting_frame = np.round(onset_frames[onset]) + (EPOCH_START_IN_MS/1000*RECORDING_FRAMERATE)
                trial_ending_frame = np.round(onset_frames[onset]) + (EPOCH_END_IN_MS/1000*RECORDING_FRAMERATE)


                #Grab this range of frames from the recording and store in epoched matrix
                epoch = video[int(trial_starting_frame):int(trial_ending_frame),:,:]
                epoched_pixels[onset,:,:] = epoch

        return epoched_pixels


def baseline_adjust_pixels(epoched_pixels):

        baseline_adjusted_epoched = np.empty(shape=epoched_pixels.shape)

        for i in range(len(epoched_pixels)):
                for j in range(len(epoched_pixels[0][0])):
                        for k in range(len(epoched_pixels[0][0])):

                                test_trace = epoched_pixels[i,:,j,k]
                                baseline_average = np.average(test_trace[0:5])
                                normalized_trace = np.subtract(test_trace,baseline_average)
                                baseline_adjusted_epoched[i,:,j,k] = normalized_trace

        return baseline_adjusted_epoched

def format_trials(baseline_adjusted_epoched,conditions):

        #Format the trials into a dict, arranged by frequency.
        #Each trace should be a nFrames by relative fluorescence array
        # format the dictionary so we get this structure:
        #     # freq_f{
        #       repetition{ 
        #           [x,x,x,x,...] }}}

        freq_dict = dict.fromkeys(np.unique(conditions[:,0]))

        # make empty dictionaries so we can index properly later
        for freq in freq_dict:
                freq_dict[freq] = {}

        # make a temporary map so we can keep track of how many repetitions of this trial we've seen
        # just going to add together the frequency and intensity to index it
        # biggest element we'll need is max(frequency)
        max_element = max(conditions[:,0]) + 10
        temp_map = [0] * max_element

        # for each trial
        for trial in range(len(conditions)):

                # trial's frequency
                f = conditions[trial,0]

                # access the map to see how many repetitions of the frequency we've already seen
                # this way we don't overwrite a trial with the same stimulus type
                num_rep = temp_map[f]+1
                temp_map[f] += 1

                # using the frequency and intensity to index our dictionary to store our trace
                freq_dict[f][num_rep] = epoched_pixels[trial,:,:,:]

        return freq_dict




#FILESTOLOAD

#Location of the tif recording you'd like to process.
folder = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/ID175_09282022_GCaMP6s_1_512x512/"

voltfile = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/VoltageRecording-09282022-1256-086/VoltageRecording-09282022-1256-086_Cycle00001_VoltageRecording_001.csv"
voltrecord = np.genfromtxt(voltfile,delimiter=',',skip_header=True)

conditions_mat = sio.loadmat("C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/ID175_09282022_1.mat")
conditions = conditions_mat["stim_data"]
conditions = conditions[1:]  #Remove the first silent stim as this corresponds to frame 0

# Opens the tif images, downsamples from 512x512 to 256x256 resolution and stores in video variable as an ndarray.  
# Note: the individual tif images must be the ONLY thing in the folder, it will try to load a tif stack as part of the variable.

#video = []
#images = [img for img in os.listdir(folder)]

#for img in images:
        #im = imread(folder+img)
        #downsamp_img = block_reduce(im,block_size=(2,2),func=np.mean)
        #video.append(downsamp_img)

#video = np.array(video)

#onset_frames = get_onset_frames(voltrecord)
#epoched_pixels = epoch_trials(video,onset_frames)
#baseline_adjust_epoched = baseline_adjust_pixels(epoched_pixels)
#trial_dict = format_trials(baseline_adjust_epoched,conditions)

#Create average response for each Trial.

with open('C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/trial_dict.pkl', 'rb') as f:
    dict = pickle.load(f)

# Take the average response of each pixel at each trial, for each frequency. 
# Create an empty dictinary in which to store the average value for each frequency.  
# Dict has the structure keys: frequency, items: array of nFrames x nPixels x nPixels (26 x 256 x 256)

average_dict = dict.fromkeys(np.unique(conditions[:,0]))
trial_array = np.empty([10,26,256,256])

for freq in dict:
        average_dict[freq] = {}
        for rep in range(1,len(dict[freq])):
                trial_array[rep-1,:,:,:] = dict[freq][rep]
        mean = np.mean(trial_array,axis=0)
        average_dict[freq] = mean

# For each frequency, create an array with frames x pixels x pixels.  For each pixel, find the max value in frames
max_value_freq = np.empty(shape=[1,256,256])
max_vals = np.empty(shape=[13,256,256])

max_dict = dict.fromkeys(np.unique(conditions[:,0]))
for freq in average_dict:
        max_dict[freq] = {}

# Create a dictionary that contains the maximum values for each frequency, for each pixel. (Freq x nPixels x nPixels)
freq_array = np.empty(shape=(np.array(list([average_dict[4364]]))).shape)
max_value_freq = np.empty(shape=[1,256,256])

for freq in average_dict:
        freq_array = np.array(list(average_dict[freq]))
        for i in range(len(freq_array[0][0])):
                for j in range(len(freq_array[0][0])):
                        max_value_freq[:,i,j] = np.amax(freq_array[:,i,j])
        print(max_value_freq)

max_array_list = []
for freq in max_dict:
        max_array_list.append(list(max_dict[freq]))
max_array = np.array(max_array_list)

#for i in range(len(max_array[:,:,0,:])):
        #for j in range(len(max_array[:,:,:,0])):
                #max_element = np.amax(max_array[:,:,i,j])
                #print(max_element)

maximal = np.amax(max_array[:,:,0,0])
test = np.array(list(max_dict[5371]))
#print(average_dict.keys())
#print(max_array_list)











# add this max value to an empty array of shape 13 x 256 x 256
# For each pixel, find the sub-level of the array 















# create a binary pickle file 
#f = open("trial_dict.pkl","wb")

# write the python object (dict) to pickle file
#pickle.dump(trial_dict,f)

# close file
#f.close()






#NOTES:
#Write the video as a tiff, may have to increase brightness in imagej to see it.  Rename 'temp.tif' to filename. 
#tifffile.imwrite('Baseline_adjust_veronica.tif',baseline_adjusted_video, photometric='minisblack')

#test_trace = epoched_pixels[0,:,150,150]
#baseline_average = np.average(test_trace[0:5])
#test_trace_baseline = test_trace[0:5]
#normalized_trace = np.subtract(test_trace,baseline_average)

#CHECK EPOCHING IS CORRECT
#onset_frames_wn = np.round(onset_frames)
#x,y = (100,100)
#pixels = video[:,x,y]
#plt.plot(pixels)
#plt.title('Pixel values for x=' + str(x) + ', y=' + str(y))
#for onset in onset_frames_wn:
        #plt.vlines(onset,4300,5000,color='black')

#plt.tight_layout()
#plt.show()

#plt.imshow(DeltaF_Fo[0,:,:],cmap='gray')
#plt.show()

#Testing if means are correct: 
#mean_0_0 = np.mean(video[:,240,100])
#print(mean_0_0)
#print(mean_pixels[240,100])


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))


