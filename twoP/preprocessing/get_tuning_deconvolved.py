"""
Script to estimate each cells' response to each frequency x intensity combination through three different methods (peak, z-score, avging)
Adds the tuning estimates into the big dictionary under the key 'tuning'
INPUT: cell_dictionary.pkl, recording_info.pkl
OUTPUT: cell_dictionary.pkl now with a key 'tuning' with response estimates to the stim types
AUTHOR: Conor Lane, Veronica Tarka, May 2022, conor.lane1995@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']
CELL_DICT_FILE_OUT = CELL_DICT_FILE
EPOCH_START_IN_MS = config['EpochStart']
FRAMERATE = config['RecordingFR']
ZSCORE_THRESHOLD = config['zscore_threshold']
n_baseline_frames =  round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

CELL_OF_INTEREST = 70

"""
Estimate cell's response to each frequency/intensity combination by taking the peak of the average trial response for a single condition type.
@param cell_traces: the contents of the 'traces' key for a single cell in the big dictionary (formatting in more details within the function)
@return tuning_curve: an nFrequency x nIntensity array where each element represents the response estimate for that frequency and intensity combination
                        where tuning_curve[0,0] is the response to the lowest frequency and smallest intensity
"""
def get_cell_tuning_by_peak(cell_traces):

    # cell_traces is a dictionary of frequencies 
    # freq_f{
    #    itsy_i{
    #       rep_r{
    #           [x,x,x,x,...]}}}
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curve = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1 # number of frames in the trial epoch to consider as baseline (pre stimulus)

    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:
        intensity_counter = 0

        # iterate through each intensity the frequency was presented at
        for intensity in cell_traces[freq]:

            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []

            # iterate through each trial of this frequency/intensity combination
            for trial in cell_traces[freq][intensity]:
                
                trace = cell_traces[freq][intensity][trial]
                all_trials_of_this_intensity.append(trace)

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)
            response = average_trial_of_this_intensity[n_baseline_frames:]

            peak_response = np.amax(response) # get the peak of the average response
            tuning_curve[frequency_counter,intensity_counter] = peak_response

            intensity_counter += 1 # go to the next intensity
        
        frequency_counter += 1 # go to the next frequency
   
    return tuning_curve

"""
Estimate cell's response to each frequency/intensity combination by taking the averaging across the average trial response for a single condition type.
@param cell_traces: the contents of the 'traces' key for a single cell in the big dictionary (formatting in more details within the function)
@return tuning_curve: an nFrequency x nIntensity array where each element represents the response estimate for that frequency and intensity combination
                        where tuning_curve[0,0] is the response to the lowest frequency and smallest intensity
"""
def get_cell_tuning_by_avg(cell_traces):

    # cell_traces is a dictionary of frequencies 
    # freq_f{
    #    itsy_i{
    #       rep_r{
    #           [x,x,x,x,...]}}}
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curve = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1 # number of frames at the start of the trial epoch to consider baseline (before trial onset)

    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:
        intensity_counter = 0

        # iterate through each intensity the frequency was presented at
        for intensity in cell_traces[freq]:

            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []
            # iterate through each trial of this frequency/intensity combination
            for trial in cell_traces[freq][intensity]:
                trace = cell_traces[freq][intensity][trial]
                all_trials_of_this_intensity.append(trace)

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)

            # take the frames of the trial occuring only after the stimulus onset
            response = average_trial_of_this_intensity[n_baseline_frames:]

            avg_response = np.average(response) # take the average of the average response
            tuning_curve[frequency_counter,intensity_counter] = avg_response

            intensity_counter += 1 # progress to the next intensity

        frequency_counter += 1 # progress to the next frequency

    return tuning_curve

def get_zscored_response(trial,baselines,n_baseline_frames):
    response = trial[n_baseline_frames:]

    baseline_mean = np.average(baselines)
    baseline_std = np.std(baselines)

    zscorer = lambda x: (x-baseline_mean)/baseline_std
    zscore_response = np.array([zscorer(xi) for xi in response])

    return zscore_response


"""
Estimate cell's response to each frequency/intensity combination by z-scoring each trial relative to 5 frames before baseline,
then taking the average of each z-scored trial for a single condition (x frequency and y intensity). 
The peak of the average z-scored trial is the response estimate.
@param cell_traces: the contents of the 'traces' key for a single cell in the big dictionary (formatting in more details within the function)
@return tuning_curve: an nFrequency x nIntensity array where each element represents the response estimate for that frequency and intensity combination
                        where tuning_curve[0,0] is the response to the lowest frequency and smallest intensity
"""
def get_cell_tuning_by_zscore(cell_traces,baselines):
    # cell_traces is a dictionary of frequencies 
    # freq_f{
    #    itsy_i{
    #       rep_r{
    #           [x,x,x,x,...]}}}
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curve = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    # get the number of frames we will use to estimate the baseline activity (frames immediately before trial onset)
    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:

        intensity_counter = 0

        # iterate through each intensity the frequency was presented at
        for intensity in cell_traces[freq]:
            # collect all the z-scored trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []

            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                
                counter+=1
                trace = cell_traces[freq][intensity][trial]

                zscore_response = get_zscored_response(trace,baselines,n_baseline_frames)
                all_trials_of_this_intensity.append(zscore_response)

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)
            

            zscored_trials_mean = np.mean(all_trials_as_np,axis=0)
          

            # average across all the trials to get a 1 x nFrames vector
            mean_response = np.mean(zscored_trials_mean)
            
          
                
            # now we grab the mean of the trace occuring AFTER the onset

           
            tuning_curve[frequency_counter,intensity_counter] = mean_response
            

            intensity_counter += 1 # move to the next intensity

        frequency_counter += 1 # move to the next frequency

    return tuning_curve



def get_cell_tuning_by_peak_zscore(cell_traces,baselines):
    # cell_traces is a dictionary of frequencies 
    # freq_f{
    #    itsy_i{
    #       rep_r{
    #           [x,x,x,x,...]}}}
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curve = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    # get the number of frames we will use to estimate the baseline activity (frames immediately before trial onset)
    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:

        intensity_counter = 0

        # iterate through each intensity the frequency was presented at
        for intensity in cell_traces[freq]:
            # collect all the z-scored trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []

            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                
                counter+=1
                trace = cell_traces[freq][intensity][trial]

                zscore_response = get_zscored_response(trace,baselines,n_baseline_frames)
                all_trials_of_this_intensity.append(zscore_response)

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)
            

            zscored_trials_mean = np.mean(all_trials_as_np,axis=0)
          

            # average across all the trials to get a 1 x nFrames vector
            peak_response = np.max(zscored_trials_mean)
            
          
                
            # now we grab the mean of the trace occuring AFTER the onset

           
            tuning_curve[frequency_counter,intensity_counter] = peak_response
            

            intensity_counter += 1 # move to the next intensity

        frequency_counter += 1 # move to the next frequency

    return tuning_curve

"""
Show a plot containing the tuning heatmaps for the first 25 active cells in this recording, each subplot title is the cell ID
@param cell_dictionary: the big cell dictionar, with key 'tuning' containing the tuning curves
@param frequencies: each frequency that was presented during the recording in ascending order
@param intensities: each intensity that was presented during the recording in ascending order
"""
def plot_tuning_curves(cell_dictionary,frequencies,intensities):
    frequencies = np.round(frequencies / 1000) # convert to kHz
    frequencies = np.array([int(f) for f in frequencies]) # round to integer

    # create the figure
    fig,axs = plt.subplots(5,5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.5,wspace=0.001)
    axs = axs.ravel() # make the axis indexing a vector rather than 2D array
    for ax,cell in zip(axs,cell_dictionary.keys()):
        cell_tuning = cell_dictionary[cell]['tuning'] # get the tuning for this cell
        im = ax.imshow(np.transpose(cell_tuning),cmap='winter',origin='lower') # plot it
        ax.set_xticks([]) # hide the axis ticks
        ax.set_yticks([])
        ax.set_title(cell) # show the cell ID as the title
        plt.colorbar(im,ax=ax) # add the color bar

    if len(frequencies)>5: # if we have a lot of frequencies
        # only show a label for every second frequency
        axs[20].set_xticks(range(0,len(frequencies),2))
        axs[20].set_xticklabels(frequencies[range(0,len(frequencies),2)])
    else:
        # otherwise show every frequency label
        axs[20].set_xticks(range(0,len(frequencies)))
        axs[20].set_xticklabels(frequencies[range(0,len(frequencies))])

    # show every intensity label
    axs[20].set_yticks(range(0,len(intensities)))
    axs[20].set_yticklabels(intensities)

    # label the axes for just one subplot
    axs[20].set_ylabel("Intensity (dB)")
    axs[20].set_xlabel("Frequency (kHz)")

    plt.show(block=False)

"""
Shows the tuning heatmap for a single cell, specified by the CELL_OF_INTEREST ID number
@param cell_tuning: the contents of the 'tuning' key for the cell
@param cell_ID: the ID of the cell to be plotted
@param frequencies: a list of frequencies presented during the recording in ascending order
@param intensities: a list of intensities presented during the recording in ascending order
"""
def plot_single_tuning_curve(cell_tuning,cell_ID,frequencies,intensities):

    fig = plt.figure()
    ax = fig.gca()

    im = plt.imshow(np.transpose(cell_tuning),cmap='winter',origin='lower')
    plt.colorbar(im)

    if len(frequencies)>5: # if we have a lot of frequencies
        # only show a label for every second frequency
        ax.set_xticks(range(0,len(frequencies),2))
        ax.set_xticklabels(frequencies[range(0,len(frequencies),2)])
    else:
        # otherwise show every frequency label
        ax.set_xticks(range(0,len(frequencies)))
        ax.set_xticklabels(frequencies[range(0,len(frequencies))])

    # show every intensity label
    ax.set_yticks(range(0,len(intensities)))
    ax.set_yticklabels(intensities)

    # label the axes
    ax.set_ylabel("Intensity (dB)")
    ax.set_xlabel("Frequency (Hz)")

    plt.title(cell_ID)
    plt.show(block=False)

"""
Plot the tuning traces for a single cell
@param cell_traces: the contents of the 'traces' key for a single cell in the big dictionary
@param n_frequencies: the total number of unique frequencies presented during the recording
@param n_intensities: the total number of unique intensities presented during the recording
@param y_limit: how tall the y axis should be for each subplot
"""
def plot_tuning_traces(cell_traces,n_frequencies,n_intensities,y_limit):

    fig,axs = plt.subplots(n_intensities,n_frequencies,sharex='col',sharey='row',figsize=(14,5))

    for row,freq in zip(range(n_frequencies),cell_traces.keys()):
        for col,itsy in zip(range(n_intensities),reversed(list(cell_traces[freq].keys()))):
            for rep in cell_traces[freq][itsy]:
                axs[col,row].plot(cell_traces[freq][itsy][rep]) # plot every trial

            # miscellaneous formatting
            axs[col,row].set_xticks([])
            axs[col,row].set_yticks([])
            if row==0:
                axs[col,row].set_ylabel(itsy) # add the intensity to the far left edge
            if col==n_intensities-1:
                axs[col,row].set_xlabel(freq) # add the frequency at the bottom
            axs[col,row].axvline(x=4,color='k',linestyle='--')
            axs[col,row].set_ylim(bottom=0,top=y_limit)
            axs[col,row].autoscale(enable=True, axis='x', tight=True)

    fig.subplots_adjust(wspace=0,hspace=0)
    fig.text(0.5,0.01,"Frequency (Hz)",va='center',ha='center')
    fig.text(0.01,0.5,"Intensity (dB)",va='center',ha='center',rotation='vertical')
    fig.suptitle(CELL_OF_INTEREST)
    plt.show(block=False)

def compute_single_baseline(cell_trace,n_baseline_frames):

    nfreq = list(cell_trace.keys())
    nInt = list(cell_trace[nfreq[0]].keys())
    nrep = list(cell_trace[nfreq[0]][nInt[0]].keys())

    trials = (np.array([[[cell_trace[i][j][k] for k in nrep] for j in nInt] for i in nfreq]))
    baselines = trials[:,:,:,:(n_baseline_frames-1)]
    
    return baselines



"""
Add tuning information to the big dictionary for each cell
@param cell_dictionary: the big cell dictionary where each cell is a key holding sub dictionaries
@return cell_dictionary: same big dictionary with new key 'tuning' holding the tuning information for each cell
"""
def get_tuning_curves(cell_dictionary):
    # for each cell, we want to add a key so our dictionary now looks like this
    # cell { 'traces' { .... }
    #        'active' = T/F
    #        'tuning' = [[x,x,x,x,x,...] 
    #                   [[x,x,x,x,x,...]...]
    # }
    # where tuning curve holds the average peak activity for that specific frequency 

    # You can use get_cell_tuning_by_zscore, get_cell_tuning_by_peak, or get_cell_tuning_by_avg here
    for cell in cell_dictionary:
        cell_dictionary[cell]['tuning'] = get_cell_tuning_by_zscore(cell_dictionary[cell]['deconvolved_traces'],compute_single_baseline(cell_dictionary[cell]['deconvolved_traces'],n_baseline_frames))
        cell_dictionary[cell]['peak_tuning'] = get_cell_tuning_by_peak_zscore(cell_dictionary[cell]['deconvolved_traces'],compute_single_baseline(cell_dictionary[cell]['deconvolved_traces'],n_baseline_frames))
    return cell_dictionary

def main():

    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dictionary = pickle.load(f)

    # load the recording info file
    with open(BASE_PATH + "recording_info.pkl","rb") as f:
        recording_info = pickle.load(f)
    
    frequencies = recording_info['frequencies']
    intensities = recording_info['intensities']
    
    cell_dictionary_with_tuning = get_tuning_curves(cell_dictionary) # add the key 'tuning' to the dictionary

    active_cell_dictionary_with_tuning = get_active_cells(cell_dictionary_with_tuning)
    
    # plot some stuff if we want
    plot_tuning_curves(active_cell_dictionary_with_tuning,frequencies,intensities)
    plot_single_tuning_curve(cell_dictionary_with_tuning[CELL_OF_INTEREST]['tuning'],CELL_OF_INTEREST,frequencies,intensities)
    plot_tuning_traces(cell_dictionary[CELL_OF_INTEREST]['deconvolved_traces'],len(frequencies),len(intensities),400)
    plt.show()
    
    # save the edited dictionary
    with open(BASE_PATH+CELL_DICT_FILE_OUT,'wb') as f:
        pickle.dump(cell_dictionary_with_tuning,f)


if __name__ == '__main__':
    main()

