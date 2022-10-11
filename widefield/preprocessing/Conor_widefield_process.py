import time
from datetime import timedelta
from timeit import repeat
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

TRIGGER_DELAY_IN_MS = 0
RECORDING_FRAMERATE = 10.01
EPOCH_START_IN_MS = -500
EPOCH_END_IN_MS = 2000
cutoff = 0.2
fs = 10.01


def load_recording(folder):

        # Opens the tif images, downsamples from 512x512 to 256x256 resolution (change block size for different downsizing) and stores in video variable as an ndarray.  
        # Note: the individual tif images must be the ONLY thing in the folder, it will try to load anything else as part of the video.

        video = []
        images = [img for img in os.listdir(folder)]

        for img in images:
                im = imread(folder+img)
                downsamp_img = block_reduce(im,block_size=(2,2),func=np.mean)
                video.append(downsamp_img)
        video = np.array(video)

        return video

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def apply_butter_highpass(video,cutoff,fs):
        for i in range(len(video[0,:,0])):
                for j in range(len(video[0,0,:])):
                        video[:,i,j] = butter_highpass_filter(video[:,i,j],cutoff,fs,order=5)
        return video





# Fits a Gaussian filter to each frame in the recording.  
def fit_multi_channel_gaussian(video):

        for frame in range(len(video)):
                video[frame,:,:] = filters.gaussian(video[frame,:,:],sigma=1,truncate=2)

        return video


# Fits a Median filter to each frame in the recording.  
def fit_median_filter(video):
        for frame in range(len(video)):
                video[frame,:,:] = median_filter(video[frame,:,:],size=3)

        return video



def convert_to_deltaF_Fo(video):
        # Takes a single pixel mean for whole pixel's trace - only really good for visualization. 
        # Take the mean of each pixel across all frames and store it in a 256x256 ndarray. 
        mean_pixels = np.empty(shape=[256,256])

        for i in range(len(video[0,:,0])):
                for j in range(len(video[0,0,:])):

                        mean = np.mean(video[:,i,j])
                        mean_pixels[i,j] = mean

        #For each frame, subtract the mean of that frame from the total recording, 
        # then divide by the mean to get (F-Fo)/Fo

        #Reshape array so can subtract mean value
        mean_pixels = mean_pixels[np.newaxis,...]
        baseline_subtracted = (np.subtract(video,mean_pixels))
        deltaF_Fo = baseline_subtracted/mean_pixels

        return deltaF_Fo


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
                freq_dict[f][num_rep] = baseline_adjusted_epoched[trial,:,:,:]

        return freq_dict

        

def trial_average(freq_dict,conditions):

        #For each pixel, create an average response across all trial repetitions, outputting an average trace for each pixel at each frequency. 
        # Creates an empty dictinary in which to store the average value for each frequency.  
        # Takes the repeteated trial values from the freq_dict and puts them into trial_array, then averages across all trials. 
        # Places the average of each frequency trial_array into the new dict. 
        # average_dict has the structure keys: frequency, items: array of nFrames x nPixels x nPixels (26 x 256 x 256)

        average_dict = freq_dict.fromkeys(np.unique(conditions[:,0]))
        trial_array = np.empty([10,26,256,256])

        for freq in freq_dict:
                average_dict[freq] = {}
                for rep in range(1,len(freq_dict[freq])):
                        trial_array[rep-1,:,:,:] = freq_dict[freq][rep]
                mean = np.mean(trial_array,axis=0)
                average_dict[freq] = mean

        return average_dict


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
                                max = np.amax(freq_array[:,i,j])
                                max_value_freq[:,i,j] = max
                max_dict[freq] = max_value_freq
        return max_dict


# For each pixel, print the value from across all frequencies that was the maximum response. 
def get_best_frequency(max_dict):

        max_array_list = []
        for freq in max_dict:
                max_array_list.append(list(max_dict[freq]))

        max_array = np.array(max_array_list)
        best_freq = np.empty(shape=[1,256,256])
        for i in range(len(max_array[1,0,:,0])):
                for j in range(len(max_array[1,0,0,:])):
                        indices = np.where(max_array[:,:,i,j] == max_array[:,:,i,j].max())
                        indices = np.array(indices[0])
                        best_freq[:,i,j] = indices[0]

        return best_freq

def color_map_rgb(value, cmap_name='hot', vmin=1, vmax=12):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    #color = matplotlib.colors.rgb2hex(rgb)
    return rgb

def convert_to_rgb(array):
        rgb_array = np.zeros([3,256,256])
        for i in range(len(array[0,:,0])):
                for j in range(len(array[0,0,:])):
                        rgb = color_map_rgb(array[:,i,j],cmap_name='hot',vmin=1,vmax=12)
                        #print(rgb)
                        rgb_array[:,i,j] = rgb[0,:3]
        return rgb_array        

'''
MAIN:

'''
#FILESTOLOAD

#Location of the tif recording to be processed.
folder = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/ID175_09282022_GCaMP6s_2_512x512/"

#Location of the voltage recording CSV file for triggers.
voltfile = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/VoltageRecording-09282022-1256-088/VoltageRecording-09282022-1256-088_Cycle00001_VoltageRecording_001.csv"
voltrecord = np.genfromtxt(voltfile,delimiter=',',skip_header=True)

#Location of the stimulus order .mat file 
conditions_mat = sio.loadmat("C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09282022_GCaMP6s_ID175/ID175_09282022_2.mat")
conditions = conditions_mat["stim_data"]
conditions = conditions[1:]  #Remove the first silent stim as this corresponds to frame 0


#Load the recording to be analyzed
video = load_recording(folder)

#Denoise the recording with a gaussian filter
#video = fit_multi_channel_gaussian(video)

#Denoise recording with median filter
#video = fit_median_filter(video)

#video = apply_butter_highpass(video,cutoff,fs)

#get onset frames of stims
onset_frames = get_onset_frames(voltrecord)

#separate recording into individual trials using onset frames 
epoched_pixels = epoch_trials(video,onset_frames)

#Baseline adjust each trial (subtract 5 pre-stimulus frames from response)
baseline_adjust_epoched = baseline_adjust_pixels(epoched_pixels)

#Format trials into a dictionary arranged by frequency
freq_dict = format_trials(baseline_adjust_epoched,conditions)

# Condense all trials for each frequency into a single average trace and store in a dictionary (keys = frequency)
average_dict = trial_average(freq_dict,conditions)

# For each pixel, find the peak of the response to stim and store it in a dict, separated by frequency.
max_dict = get_max_response(average_dict)

# Find the frequency elicting the maximum response for each pixel. Arranged in an image-shaped array with frequency values converted to 1-N integers 
# (lowest freq = 0) 
best_frequency = get_best_frequency(max_dict)

#best_frequency_gaussian = gaussian_filter(best_frequency,sigma=1)

best_frequency_median = median_filter(best_frequency,size=3)

# Convert the best frequency array to an RGB format. Transpose so it can be plotted as an image.
#BF_map = convert_to_rgb(best_frequency)
#BF_map = np.transpose(BF_map, (1,2,0))


fig, ax = plt.subplots()
data = np.squeeze(best_frequency_median)
cax = ax.imshow(data, cmap=cm.jet)
ax.set_title('Map with median Filter')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, ticks=[2, 4, 6, 8, 10, 12])
cbar.ax.set_yticklabels(['4364', '6612', '10020', '15184', '23009', '42922'])  # vertically oriented colorbar
plt.show()


"""
Redefine trial activity as z-scores relative to the baseline frames immediately preceding the stimulus
@param trial: nFrames x 1 vector of dF/F values over the trial
@param n_baseline_frames: the number of frames included in the trial epoch which preceded the stimulus
@return zscore_response: the frames occuring after the stimulus onset now defined as z-scores relative to pre-stim baseline
"""


# def get_zscored_response(trial,n_baseline_frames):
#     baseline = trial[:n_baseline_frames]
#     response = trial[n_baseline_frames:]

#     baseline_mean = np.average(baseline)
#     baseline_std = np.std(baseline)

#     zscorer = lambda x: (x-baseline_mean)/baseline_std
#     zscore_response = np.array([zscorer(xi) for xi in response])

#     return zscore_response


#CHECK EPOCHING IS CORRECT
#onset_frames_wn = np.round(onset_frames)
#x,y = (150,150)
#pixels = video[:,x,y]
#plt.plot(pixels)
#plt.title('Pixel values for x=' + str(x) + ', y=' + str(y))
#for onset in onset_frames_wn:
     #   plt.vlines(onset,0,50,color='black')

#plt.tight_layout()
#plt.show()
















#with open('C:/Users/Conor/2Psinapod/2Psinapod/widefield/preprocessing/average_dict.pkl', 'rb') as f:
      #  average_dict = pickle.load(f)

#average_dict.pop('1000', None)

#n_baseline_frames = 5

#zscore_dict = dict.fromkeys(np.unique(conditions[:,0]))
#zscore_dict.pop('1000', None)

#for freq in average_dict:
       # zscore_dict[freq] = {}
      #  freq_array = average_dict[freq]
       # zscore_array = np.empty([21,256,256])
       # for i in range(len(freq_array[0,:,0])):
              #  for j in range(len(freq_array[0,0,:])):
                      #  zscore_array[:,i,j] = get_zscored_response(freq_array[:,i,j],n_baseline_frames)
       # zscore_dict[freq] = zscore_array

#freq_array = average_dict[1000]
#print(np.mean(freq_array[:,150,150]))



# test_array = np.array(zscore_dict[5371])       
# print(test_array[:,150,150])




# create a binary pickle file 
#f = open("average_dict.pkl","wb")

# write the python object (dict) to pickle file
#pickle.dump(average_dict,f)

# close file
#f.close()





#normalize vector across all frequencies - divide by the sum
# set a threshold e.g. 0.5 for how much of the response is that frequency
# plot a histogram to find tuning strength
# create an array 12 x 256 x 256 - smooth along the axis of each of the 12 "images"
#cm jet





#with open('C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/trial_dict.pkl', 'rb') as f:
    #dict = pickle.load(f)


















#NOTES:
#Write the video as a tiff, may have to increase brightness in imagej to see it.  Rename 'temp.tif' to filename. 
#tifffile.imwrite('Baseline_adjust_veronica.tif',baseline_adjusted_video, photometric='minisblack')

#test_trace = epoched_pixels[0,:,150,150]
#baseline_average = np.average(test_trace[0:5])
#test_trace_baseline = test_trace[0:5]
#normalized_trace = np.subtract(test_trace,baseline_average)



#plt.imshow(DeltaF_Fo[0,:,:],cmap='gray')
#plt.show()

#Testing if means are correct: 
#mean_0_0 = np.mean(video[:,240,100])
#print(mean_0_0)
#print(mean_pixels[240,100])


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))


