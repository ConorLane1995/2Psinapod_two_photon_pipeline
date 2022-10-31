from cmath import nan
import time
from datetime import timedelta
from timeit import repeat
from tkinter import Y
start_time = time.monotonic()

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imread
from skimage.measure import block_reduce
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import tifffile
import scipy.io as sio
import pickle
from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy import signal
from matplotlib.colors import ListedColormap

TRIGGER_DELAY_IN_MS = 0
RECORDING_FRAMERATE = 10
EPOCH_START_IN_MS = -500
EPOCH_END_IN_MS = 2000
n_baseline_frames = 5
# zscore_threshold = 50
cutoff = 0.2
fs = 10
order = 5

'''
Load the tiff stack of the recording as a single 3D array and downsample it from 512x512 to 256x256 (if recording is larger than 512x512, change block size).   
Note: The tiff stack must be the only thing in the folder.  It will try to load other items into the array.
@Param: Name of folder (paste path into FILESTOLOAD section)
Return: (N_frames x N_pixels x N_pixels) numpy array.
'''
def load_recording(folder):

        video = []
        images = [img for img in os.listdir(folder)]

        for img in images:
                im = imread(folder+img)
                downsamp_img = block_reduce(im,block_size=(2,2),func=np.mean)
                video.append(downsamp_img)
        video = np.array(video)

        return video
'''
Applies a Butterworth high pass filter to the full time-course of each pixel, to remove slow fluctuations in the signal. 
@Param: Cutoff - The frequency below which activity will be filtered out of the signal. 
@Param: fs - framerate of the recording in Hz
@Param: The order of the filter. For 'bandpass' and 'bandstop' filters, the resulting order of the final second-order 
@Param: Video - 
sections ('sos') matrix is 2*N, with N the number of biquad sections of the desired system.
'''

def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(video, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, video)
    return y

def apply_butter_highpass(video,cutoff,fs):
        for i in range(len(video[0,:,0])):
                for j in range(len(video[0,0,:])):
                        video[:,i,j] = butter_highpass_filter(video[:,i,j],cutoff,fs,order)
        return video


# Fits a Gaussian filter to each frame in the recording.  
def fit_multi_channel_gaussian(video):

        for frame in range(len(video)):
                video[frame,:,:] = filters.gaussian(video[frame,:,:],sigma=1,truncate=2)

        return video


# Fits a Median filter to each frame in the recording.  
def fit_median_filter(video,size):
        for frame in range(len(video)):
                video[frame,:,:] = median_filter(video[frame,:,:],size=size)

        return video

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

    #Remove first three triggers, corresponding to start at frame zero, 
    onset_frames_at_recording_fr = onset_frames_at_recording_fr[3:]

    return onset_frames_at_recording_fr


def epoch_trials(video,onset_frames):

        # Get length of trial in seconds
        trial_length_in_ms = EPOCH_END_IN_MS - EPOCH_START_IN_MS # this gives us length in ms
        trial_length_in_sec = trial_length_in_ms/1000 # now we have it in second

        # Convert this to length in frames
        trial_length_in_frames = int(trial_length_in_sec * RECORDING_FRAMERATE) # s * f/s = f

        # Initialize an array to store the epoched traces
        # nTrials x nFrames x nPixels x nPixels

        epoched_pixels = np.zeros((len(onset_frames),(trial_length_in_frames), len(video[0,:,0]), len(video[0,0,:])))

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

'''
Normalize each trial to it's local pre-stimulus baseline by subtracting the mean of the pre-stim from each timepoint in the trial. 
@Param epoched pixels =  N_trials x N_frames x N_pixels x N_pixels array.
@param n_baseline_frames = The number of pre-stimulus baseline frames to use in the normalization. 
@Returns:  Ntrials x N_frames x N_pixels x N_pixels array of baseline adjusted trials. e.g. [0,:,0,0] is the normalized trace of the 
first trial at pixel 0,0. 
'''

def baseline_adjust_pixels(epoched_pixels,n_baseline_frames):
        # Create an empty array to store the baseline adjusted trials in. Same shape as epoched pixels.
        baseline_adjusted_epoched = np.empty(shape=epoched_pixels.shape)

        # Iterate through the trials (i) and each x any y pixel coordinate (j and K)
        for i in range(len(epoched_pixels)):
                for j in range(len(epoched_pixels[0][0])):
                        for k in range(len(epoched_pixels[0][0])):

                                # Extract the specific trial to be normalized
                                test_trace = epoched_pixels[i,:,j,k]
                                # compute the average of the number of baseline frames
                                baseline_average = np.average(test_trace[0:n_baseline_frames])
                                normalized_trace = np.subtract(test_trace,baseline_average)
                                baseline_adjusted_epoched[i,:,j,k] = normalized_trace

        return baseline_adjusted_epoched


def single_baseline_adjust(epoched_pixels,n_baseline_frames):

        baseline_mean_array = np.empty([1,256,256])
        single_baseline_epoched = np.empty(shape=epoched_pixels.shape)

        for i in range(len(epoched_pixels[0,0,:,0])):
                for j in range(len(epoched_pixels[0,0,0,:])):
                                baseline = np.mean(epoched_pixels[:,:n_baseline_frames,i,j])
                                baseline_mean_array[:,i,j] = baseline
        

        for i in range(len(epoched_pixels[0,0,:,0])):
                for j in range(len(epoched_pixels[0,0,0,:])):
                        for k in range(len(epoched_pixels[:,0,0,0])):
                                trace = epoched_pixels[k,:,i,j]
                                single_baseline_epoched[k,:,i,j] = np.subtract(trace,baseline_mean_array[:,i,j])
        return single_baseline_epoched


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
                freq_dict[f][num_rep] = baseline_adjusted_epoched[trial,:,:,:]

        return freq_dict

        

def trial_average(freq_dict,conditions):

        #For each pixel, create an average response across all trial repetitions, outputting an average trace for each pixel at each frequency. 
        # Creates an empty dictinary in which to store the average value for each frequency.  
        # Takes the repeteated trial values from the freq_dict and puts them into trial_array, then averages across all trials. 
        # Places the average of each frequency trial_array into the new dict. 
        # average_dict has the structure keys: frequency, items: array of nFrames x nPixels x nPixels (26 x 256 x 256)

        average_dict = freq_dict.fromkeys(np.unique(conditions[:,0]))
        trial_array = np.empty([10,25,256,256])

        for freq in freq_dict:
                average_dict[freq] = {}
                for rep in range(1,len(freq_dict[freq])):
                        trial_array[rep-1,:,:,:] = freq_dict[freq][rep]
                mean = np.mean(trial_array,axis=0)
                average_dict[freq] = mean

        return average_dict

def get_zscored_response(trial,n_baseline_frames):
    baseline = trial[:n_baseline_frames]
    #response = trial[n_baseline_frames:]

    baseline_mean = np.average(baseline)
    baseline_std = np.std(baseline)

    zscorer = lambda x: (x-baseline_mean)/baseline_std
    zscore_response = np.array([zscorer(xi) for xi in trial])

    return zscore_response

def zscore_and_average(freq_dict,conditions):

        zscore_dict = dict.fromkeys(np.unique(conditions[:,0]))

        for freq in freq_dict:
                freq_array = np.empty([len(freq_dict[freq]),25,256,256])
                zscore_array = np.empty([len(freq_dict[freq]),25,256,256])
                zscore_dict[freq] = {}

                for rep in range(1,len(freq_dict[freq])):
                        freq_array[rep-1,:,:,:] = freq_dict[freq][rep]
                        for i in range(len(freq_array[0,0,:,0])):
                                for j in range(len(freq_array[0,0,0,:])):
                                        zscore_array[rep-1,:,i,j] = get_zscored_response(freq_array[rep-1,:,i,j],n_baseline_frames)
                zscore_array_mean = np.mean(zscore_array,axis=0)
                zscore_dict[freq] = zscore_array_mean                                                                                                                                                                                                                                                                                                                                                                       

        return zscore_dict

def zscore_and_median(freq_dict,conditions):

        median_zscore_dict = dict.fromkeys(np.unique(conditions[:,0]))

        for freq in freq_dict:
                freq_array = np.empty([len(freq_dict[freq]),25,256,256])
                zscore_array = np.empty([len(freq_dict[freq]),25,256,256])
                ave_zscore_array = np.empty([len(freq_dict[freq]),256,256])
                median_zscore_array = np.empty([1,256,256])
                median_zscore_dict[freq] = {}

                for rep in range(1,len(freq_dict[freq])):
                        freq_array[rep-1,:,:,:] = freq_dict[freq][rep]
                        for i in range(len(freq_array[0,0,:,0])):
                                for j in range(len(freq_array[0,0,0,:])):
                                        zscore_array[rep-1,:,i,j] = get_zscored_response(freq_array[rep-1,:,i,j],n_baseline_frames)
                                        ave_zscore_array[rep-1,i,j] = np.mean(zscore_array[rep-1,7:15,i,j])
                for i in range(len(ave_zscore_array[0,:,0])):
                        for j in range(len(ave_zscore_array[0,0,:])):
                                median_zscore_array[:,i,j] = np.median(ave_zscore_array[:,i,j])

                median_zscore_dict[freq] = median_zscore_array

        return median_zscore_dict


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_max_response(average_dict):
        # For each frequency, create an array with frames x pixels x pixels.  For each pixel, find the max value in frames
        max_dict = dict.fromkeys(np.unique(conditions[:,0]))
        # Create a dictionary that contains the maximum values for each frequency, for each pixel. (Freq x nPixels x nPixels)
        for freq in average_dict:
                max_dict[freq] = {}
                freq_array = np.array(list(average_dict[freq]))
                max_value_freq = np.empty(shape=[1,256,256])
                for i in range(len(freq_array[0][0])):
                        for j in range(len(freq_array[0][0])):
                                max = np.amax(moving_average(freq_array[5:13,i,j],3))
                                max_value_freq[:,i,j] = max
                max_dict[freq] = max_value_freq
        return max_dict

def get_responsive_pixels(max_dict,conditions,zscore_threshold):
        max_dict_responsiveorno = dict.fromkeys(np.unique(conditions[:,0]))

        for freq in max_dict:
                max_dict_array = max_dict[freq]
                max_dict_responsiveorno[freq] = {}
                max_array_isresponsive = np.empty([1,256,256])
                for i in range(len(max_dict_array[0,:,0])):
                        for j in range(len(max_dict_array[0,0,:])):
                                if max_dict_array[:,i,j] > zscore_threshold:
                                        max_array_isresponsive[:,i,j] = 1
                                else:
                                        max_array_isresponsive[:,i,j] = 0
                max_dict_responsiveorno[freq] = max_array_isresponsive

        return max_dict_responsiveorno

def get_only_significant_max(max_dict,max_dict_responsiveorno,conditions):

        max_dict_significant = dict.fromkeys(np.unique(conditions[:,0]))

        for freq in max_dict:
                max_dict_significant[freq] = {}
                freq_array = max_dict[freq]
                responsiveorno = max_dict_responsiveorno[freq]
                freq_responsive = np.empty([1,256,256])
                for i in range(len(freq_array[0,:,0])):
                        for j in range(len(freq_array[0,0,:])):
                                if responsiveorno[:,i,j] == 1:
                                        freq_responsive[:,i,j] = freq_array[:,i,j]
                                else:
                                        freq_responsive[:,i,j] = 0
                max_dict_significant[freq] = freq_responsive

        return max_dict_significant


# For each pixel, print the value from across all frequencies that was the maximum response. 
def get_best_frequency(max_dict_significant):

        max_array_list = []
        for freq in max_dict_significant:
                max_array_list.append(list(max_dict_significant[freq]))

        max_array = np.array(max_array_list)
        best_freq = np.empty(shape=[1,256,256])
        for i in range(len(max_array[1,0,:,0])):
                for j in range(len(max_array[1,0,0,:])):
                        indices = np.where(max_array[:,:,i,j] == max_array[:,:,i,j].max())
                        indices = np.array(indices[0])
                        if len(indices) == 1:
                                best_freq[:,i,j] = indices[0]
                        else:
                                best_freq[:,i,j] = float('nan')

        return best_freq   



'''
MAIN:

'''
#FILESTOLOAD

#Location of the tif recording to be processed.
folder = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/27102022_GCaMP6s_ID173/ID173_27102022_GCaMP6s_2/"

#Location of the voltage recording CSV file for triggers.
voltfile = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/27102022_GCaMP6s_ID173/VoltageRecording-10272022-1526-129_Cycle00001_VoltageRecording_001.csv"
voltrecord = np.genfromtxt(voltfile,delimiter=',',skip_header=True)

#Location of the stimulus order .mat file 
conditions_mat = sio.loadmat("C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/27102022_GCaMP6s_ID173/ID173_27102022_2.mat")
conditions = conditions_mat["stim_data"]
conditions = conditions[3:]  #Remove the first silent stim as this corresponds to frame 0


#Load the recording to be analyzed
video = load_recording(folder)

# # # # #video = apply_butter_highpass(video,cutoff,fs)

# # # # # #Denoise the recording with a gaussian filter
# # # # # #video = fit_multi_channel_gaussian(video)

# # # # # #Denoise recording with median filter
# # # # # #video = fit_median_filter(video,3)

# #get onset frames of stims
onset_frames = get_onset_frames(voltrecord)

# #separate recording into individual trials using onset frames 
epoched_pixels = epoch_trials(video,onset_frames)                                                                                               

# # #Baseline adjust each trial (subtract 5 pre-stimulus frames from response)
baseline_adjusted_epoched = baseline_adjust_pixels(epoched_pixels,n_baseline_frames)

# # # # # # # # #Baseline adjust each trial using a single baseline per pixel
# # # # # # # # #baseline_adjusted_epoched = single_baseline_adjust(epoched_pixels,n_baseline_frames)

# #Format trials into a dictionary arranged by frequency
freq_dict = format_trials(baseline_adjusted_epoched,conditions)

# # # # # # # # # #Convert each individual trial rep into a z-score and average all ten repeats of a single trial. 
# # # # # mean_zscore_dict = zscore_and_average(freq_dict,conditions)

# Zscore the individual trials, and return a dict of single median value of the trial period, for each pixel.    
median_zscore_dict = zscore_and_median(freq_dict,conditions)

# # # # # # Condense all trials for each frequency into a single average trace and store in a dictionary (keys = frequency)
# # # #average_dict = trial_average(freq_dict,conditions)

# # # For each pixel, find the peak of the response to stim and store it in a dict where keys are frequency. 
# max_dict = get_max_response(mean_zscore_dict)

# # #  Returns a dictionary in the same shape as max_dict or median_zscore_dict, where the corresponding values in array space are 
# # #  represented as 1 if they are above the zscore threshold (#SD's above baseline) and 0 if not. 
# dict_responsiveorno = get_responsive_pixels(max_dict,conditions,2)

# # # # # #  Returns a dictionary (same structure as max_dict) where only the values above the pre-set response threshold (#SD's from baseline) are retained.  
# # # # # #  All sub-threshold values are returned as 0. 
# dict_significant = get_only_significant_max(max_dict,dict_responsiveorno,conditions)

# # # # # # # Find the frequency elicting the maximum response for each pixel. Arranged in an image-shaped array with frequency values converted to 0-N integers. 
# # # # # # # NaN values in the array will be interpreted as empty space (white). 
# best_frequency = get_best_frequency(dict_significant)

#best_frequency = gaussian_filter(best_frequency,sigma=1,truncate=4)

#best_frequency = median_filter(best_frequency,size=2)

# # PLOT ALL FREQUENCIES IN ONE TONOTOPIC MAP
# fig, ax = plt.subplots()
# data = np.squeeze(best_frequency)
# cax = ax.imshow(data,cmap=cm.jet)
# ax.set_title('Map with median zscore')
# # Add colorbar, make sure to specify tick locations to match desired ticklabels
# cbar = fig.colorbar(cax, ticks=[0, 2, 4, 6, 8, 11])
# cbar.ax.set_yticklabels(['4364', '6612', '10020', '15184', '23009', '42922'])  # vertically oriented colorbar
# plt.show()


# # PLOT EACH FREQUENCY AS A SEPARATE SUBPLOT
# fig,axes = plt.subplots(nrows=3, ncols=4, constrained_layout=True)
# axes = axes.ravel()
# for i, (key, value) in enumerate(dict_responsiveorno.items()):
#         axes[i].imshow(np.squeeze(value))
#         axes[i].title.set_text(key)
# plt.suptitle('ID173_27102022 Recording 1, 5:13 zscore threshold 2')
# plt.show()

## PLOT RAW TRACES FROM FREQ_DICT, FOR A SPECIFIC PIXEL
# frequency = 6612
# x,y = (50,50)
# freq_dict_array = np.empty([len(freq_dict[frequency]),25,256,256])
# for rep in freq_dict[frequency]:
#         freq_dict_array[rep-1,:,:,:] = freq_dict[frequency][rep]

# plt.plot(np.transpose(freq_dict_array[:,:,y,x]))  ###NOTE:  x and y are reversed because indexing the array (row then column) is the opposite of how the image pixels are arranged.
# plt.title(str(frequency) + ' Hz' + ' x = '+ str(x) +  ' y= '+ str(y))
# plt.legend([0,1,2,3,4,5,6,7,8,9])
# plt.show()


#Round median dict to 1 DP and plot as amplitude map.

rounded = {key : np.around(median_zscore_dict[key], 1) for key in median_zscore_dict}

fig,axes = plt.subplots(nrows=3, ncols=4, constrained_layout=True)
axes = axes.ravel()
for i, (key, value) in enumerate(rounded.items()):
        axes[i].imshow(np.squeeze(value))
        axes[i].title.set_text(key)
plt.suptitle('ID173_27102022 Recording 2, 7:15 Median Amp. Map')
plt.show()



# PLOT AVERAGED, Z-SCORED TRACES FOR ALL FREQUENCIES, FROM MEAN_ZSCORE_DICT
# x,y = (240,25)
# print(best_frequency[:,y,x])  ###NOTE:  x and y are reversed because indexing the array (row then column) is the opposite of how the image pixels are arranged.
# mean_zscore_list = []
# for frequency in mean_zscore_dict:
#         mean_zscore_list.append(list(mean_zscore_dict[frequency]))
# mean_zscore_array = np.array(mean_zscore_list)
# plot_array = mean_zscore_array[:,:,y,x] 
# plot_array = np.swapaxes(plot_array,0,1)

# plt.plot(plot_array)
# plt.title('Zscore, x = ' + str(x) + ', y = ' + str(y))
# plt.legend(['4364', '5371', '6612', '8140', '10020', '12335', '15184', '18691', '23009', '28324', '34867', '42922'],loc='upper right')
# plt.show()

# ##CHECK EPOCHING IS CORRECT
# onset_frames_wn = np.round(onset_frames)
# x,y = (50,50)
# pixels = video[:,x,y]
# plt.plot(pixels)
# plt.title('Pixel values for x=' + str(x) + ', y=' + str(y))
# for onset in onset_frames_wn:
#        plt.vlines(onset,-500,500,color='black')

# plt.tight_layout()
# plt.show()

#normalize vector across all frequencies - divide by the sum
# set a threshold e.g. 0.5 for how much of the response is that frequency
# plot a histogram to find tuning strength
# create an array 12 x 256 x 256 - smooth along the axis of each of the 12 "images"

# with open('C:/Users/Conor/2Psinapod/2Psinapod/widefield/preprocessing/max_dict.pkl', 'rb') as f:
#         max_dict = pickle.load(f)
# del max_dict[1000]



#NOTES:
#Write the video as a tiff, may have to increase brightness in imagej to see it.  Rename 'temp.tif' to filename. 
#tifffile.imwrite('Baseline_adjust_veronica.tif',baseline_adjusted_video, photometric='minisblack')

# >>> import numpy as np
# >>> arr = np.array(img)
# >>> arr[arr < 10] = 0
# >>> img.putdata(arr)

# #create a binary pickle file 
# f = open("median_zscore_dict.pkl","wb")

# #write the python object (dict) to pickle file
# pickle.dump(median_zscore_dict,f)

# #close file
# f.close()

# maxime = {key: np.amax(median_zscore_dict[key]) for key in median_zscore_dict}
# minime = {key: np.amax(median_zscore_dict[key]) for key in median_zscore_dict}
# for k, v in maxime.items():
#     print(k, v)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))