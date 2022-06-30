from cmath import sqrt
from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import sys
import json
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
from utils import get_active_cells
from scipy.stats import zscore

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
cell_dictionary_file = config['AnalysisFile']
cell_dictionary_file_out = cell_dictionary_file
EPOCH_START_IN_MS = config['EpochStart']
EPOCH_END_IN_MS = config['EpochEnd'] # time after trial onset included in the epoch
FRAMERATE = config['RecordingFR']

CELL_OF_INTEREST = 1

def get_cell_tuning_by_peak(cell_traces,plot_TF):

    if plot_TF:
        fig,axs = plt.subplots(7,9)
        # axs = axs.ravel()

    # cell_traces is a dictionary of frequencies 
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curves = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

    plot_coln_counter = 0
    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:

        intensity_counter = 0

        # find the number of intensities we presented at
        n_intensities = len(cell_traces[freq].keys())

        # make a temporary vector to append to the tuning curve at the end of this loop
        # we will fill one n_intensities length column of the 2D matrix we are returning
        activation_per_intensity = np.empty((n_intensities,1))

        # iterate through each intensity the frequency was presented at
        plot_row_counter = 0
        for intensity in cell_traces[freq]:

            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []
            n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1
            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                
                # if plot_TF:
                #     plt.plot(cell_traces[freq][intensity][trial][n_baseline_frames:])

                counter+=1
                trace = cell_traces[freq][intensity][trial]
                # print(trace)
                # input()
                # baseline = trace[0:n_baseline_frames]
                # baseline_mean = np.average(baseline)
                # baseline_std = np.std(baseline)

                # zscorer = lambda x: (x-baseline_mean)/baseline_std

                # response = trace[n_baseline_frames:]
                # zscore_response = np.array([zscorer(xi) for xi in response])
                all_trials_of_this_intensity.append(trace)

            # plt.show()

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)

            # now we grab the peak of the trace occuring AFTER the onset
            
            # baseline = average_trial_of_this_intensity[0:n_baseline_frames]
            # baseline_mean = np.average(baseline)
            # baseline_std = np.std(baseline)

            # zscorer = lambda x: (x-baseline_mean)/baseline_std

            # response = average_trial_of_this_intensity[n_baseline_frames:]
            # zscore_response = np.array([zscorer(xi) for xi in response])
            response = average_trial_of_this_intensity[n_baseline_frames:]

            if plot_TF:
                error = []
                for timepoint in range(len(all_trials_as_np[0])):
                    if timepoint<n_baseline_frames:
                        continue

                    timepoint_std = np.std(all_trials_as_np[:,timepoint])
                    timepoint_se = timepoint_std/sqrt(len(all_trials_as_np[:,timepoint]))
                    error.append(timepoint_se)

            if plot_TF:
                # print(len(error))
                # print(len(response))
                axs[6-plot_row_counter,plot_coln_counter].plot(np.transpose(all_trials_as_np))
                axs[6-plot_row_counter,plot_coln_counter].axvline(x=4,color='k',linestyle='--')
                # axs[plot_row_counter,plot_coln_counter].plot(response)
                # axs[plot_row_counter,plot_coln_counter].fill_between(range(len(response)),response-error,response+error,alpha=0.5)
                axs[6-plot_row_counter,plot_coln_counter].xaxis.set_visible(False)
                axs[6-plot_row_counter,plot_coln_counter].yaxis.set_visible(False)
                axs[6-plot_row_counter,plot_coln_counter].autoscale(enable=True, axis='x', tight=True)
                axs[6-plot_row_counter,plot_coln_counter].set_ylim(bottom=0,top=500)
                # axs[plot_row_counter,plot_coln_counter].title.set_text(intensity)

            # zscore_response = zscore(response)
            peak_response = np.amax(response)
            # peak_response = np.amax(response)
            # peak_response = np.trapz(response)
            tuning_curves[frequency_counter,intensity_counter] = peak_response

            intensity_counter += 1
            # print(freq)
            # print(intensity)
            plot_row_counter += 1

        plot_coln_counter += 1
        frequency_counter += 1
    
    if plot_TF:
        fig.subplots_adjust(wspace=0,hspace=0)
        plt.show()
    return tuning_curves

def get_cell_tuning_by_avg(cell_traces,plot_TF):

    if plot_TF:
        fig,axs = plt.subplots(7,9)
        # axs = axs.ravel()

    # cell_traces is a dictionary of frequencies 
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curves = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

    plot_coln_counter = 0
    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:

        intensity_counter = 0

        # find the number of intensities we presented at
        n_intensities = len(cell_traces[freq].keys())

        # make a temporary vector to append to the tuning curve at the end of this loop
        # we will fill one n_intensities length column of the 2D matrix we are returning
        activation_per_intensity = np.empty((n_intensities,1))

        # iterate through each intensity the frequency was presented at
        plot_row_counter = 0
        for intensity in cell_traces[freq]:

            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []
            n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1
            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                
                # if plot_TF:
                #     plt.plot(cell_traces[freq][intensity][trial][n_baseline_frames:])

                counter+=1
                trace = cell_traces[freq][intensity][trial]
                # print(trace)
                # input()
                # baseline = trace[0:n_baseline_frames]
                # baseline_mean = np.average(baseline)
                # baseline_std = np.std(baseline)

                # zscorer = lambda x: (x-baseline_mean)/baseline_std

                # response = trace[n_baseline_frames:]
                # zscore_response = np.array([zscorer(xi) for xi in response])
                all_trials_of_this_intensity.append(trace)

            # plt.show()

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)

            # now we grab the peak of the trace occuring AFTER the onset
            
            # baseline = average_trial_of_this_intensity[0:n_baseline_frames]
            # baseline_mean = np.average(baseline)
            # baseline_std = np.std(baseline)

            # zscorer = lambda x: (x-baseline_mean)/baseline_std

            # response = average_trial_of_this_intensity[n_baseline_frames:]
            # zscore_response = np.array([zscorer(xi) for xi in response])
            response = average_trial_of_this_intensity[n_baseline_frames:]

            if plot_TF:
                error = []
                for timepoint in range(len(all_trials_as_np[0])):
                    if timepoint<n_baseline_frames:
                        continue

                    timepoint_std = np.std(all_trials_as_np[:,timepoint])
                    timepoint_se = timepoint_std/sqrt(len(all_trials_as_np[:,timepoint]))
                    error.append(timepoint_se)

            if plot_TF:
                # print(len(error))
                # print(len(response))
                axs[6-plot_row_counter,plot_coln_counter].plot(np.transpose(all_trials_as_np))
                axs[6-plot_row_counter,plot_coln_counter].axvline(x=4,color='k',linestyle='--')
                # axs[plot_row_counter,plot_coln_counter].plot(response)
                # axs[plot_row_counter,plot_coln_counter].fill_between(range(len(response)),response-error,response+error,alpha=0.5)
                axs[6-plot_row_counter,plot_coln_counter].xaxis.set_visible(False)
                axs[6-plot_row_counter,plot_coln_counter].yaxis.set_visible(False)
                axs[6-plot_row_counter,plot_coln_counter].autoscale(enable=True, axis='x', tight=True)
                axs[6-plot_row_counter,plot_coln_counter].set_ylim(bottom=0,top=500)
                # axs[plot_row_counter,plot_coln_counter].title.set_text(intensity)

            # zscore_response = zscore(response)
            avg_response = np.average(response)
            # peak_response = np.amax(response)
            # peak_response = np.trapz(response)
            tuning_curves[frequency_counter,intensity_counter] = avg_response

            intensity_counter += 1
            # print(freq)
            # print(intensity)
            plot_row_counter += 1

        plot_coln_counter += 1
        frequency_counter += 1
    
    if plot_TF:
        fig.subplots_adjust(wspace=0,hspace=0)
        plt.show()
    return tuning_curves

def get_cell_tuning_by_zscore(cell_traces,plot_TF):

    if plot_TF:
        fig,axs = plt.subplots(4,12)
        # axs = axs.ravel()

    # cell_traces is a dictionary of frequencies 
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curves = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1

    plot_coln_counter = 0
    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:
        intensity_counter = 0

        # find the number of intensities we presented at
        n_intensities = len(cell_traces[freq].keys())

        # make a temporary vector to append to the tuning curve at the end of this loop
        # we will fill one n_intensities length column of the 2D matrix we are returning
        activation_per_intensity = np.empty((n_intensities,1))

        # iterate through each intensity the frequency was presented at
        plot_row_counter = 0
        for intensity in cell_traces[freq]:
            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []
            n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1
            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                
                # if plot_TF:
                    # plt.plot(cell_traces[freq][intensity][trial])#[n_baseline_frames:])

                counter+=1
                trace = cell_traces[freq][intensity][trial]

                baseline = trace[0:n_baseline_frames]
                baseline_mean = np.average(baseline)
                baseline_std = np.std(baseline)

                if baseline_std!=0:
                    zscorer = lambda x: (x-baseline_mean)#/baseline_std
                else:
                    zscorer = lambda x: x


                response = trace[n_baseline_frames:]
                zscore_response = np.array([zscorer(xi) for xi in response])
                # print(zscore_response)
                # input()
                all_trials_of_this_intensity.append(zscore_response)

            # plt.show()

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)

            # now we grab the peak of the trace occuring AFTER the onset
            
            # baseline = average_trial_of_this_intensity[0:n_baseline_frames]
            # baseline_mean = np.average(baseline)
            # baseline_std = np.std(baseline)

            # zscorer = lambda x: (x-baseline_mean)/baseline_std

            # response = average_trial_of_this_intensity[n_baseline_frames:]
            # zscore_response = np.array([zscorer(xi) for xi in response])

            response = average_trial_of_this_intensity#[n_baseline_frames:]

            if plot_TF:
                error = []
                for timepoint in range(len(all_trials_as_np[0])):
                    if timepoint<n_baseline_frames:
                        continue

                    timepoint_std = np.std(all_trials_as_np[:,timepoint])
                    timepoint_se = timepoint_std/sqrt(len(all_trials_as_np[:,timepoint]))
                    error.append(timepoint_se)

            if plot_TF:
                # print(len(error))
                # print(len(response))
                axs[plot_row_counter,plot_coln_counter].plot(np.transpose(all_trials_as_np))
                # axs[5-plot_row_counter,plot_coln_counter].axvline(x=4,color='k')
                # axs[plot_row_counter,plot_coln_counter].plot(response)
                # axs[plot_row_counter,plot_coln_counter].fill_between(range(len(response)),response-error,response+error,alpha=0.5)
                axs[plot_row_counter,plot_coln_counter].xaxis.set_visible(False)
                axs[plot_row_counter,plot_coln_counter].yaxis.set_visible(False)
                axs[plot_row_counter,plot_coln_counter].autoscale(enable=True, axis='x', tight=True)
                axs[plot_row_counter,plot_coln_counter].set_ylim(bottom=0,top=150)
                # axs[plot_row_counter,plot_coln_counter].title.set_text(intensity)

            # zscore_response = zscore(response)
            peak_response = np.amax(response)
            # peak_response = np.amax(response)
            # peak_response = np.trapz(response)
            tuning_curves[frequency_counter,intensity_counter] = peak_response

            intensity_counter += 1
            # print(freq)
            # print(intensity)
            plot_row_counter += 1

        plot_coln_counter += 1
        frequency_counter += 1
    
    if plot_TF:
        fig.subplots_adjust(wspace=0,hspace=0)
        plt.show()
    return tuning_curves

def get_cell_tuning_by_area(cell_traces):
        # cell_traces is a dictionary of frequencies 
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x 2 (first column is the frequency, second column is the activity)
    tuning_curves = np.empty((len(cell_traces),2))

    counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:
        tuning_curves[counter,0] = freq
        # print(tuning_curves[counter,0])

        # now we need to get the peak intensity of the average of all the trials for that one frequency 
        n_samples = 0
        n_trials = 0
        for intensity in cell_traces[freq]:
            for repetition in cell_traces[freq][intensity]:
                if n_trials == 0:
                    n_samples = len(cell_traces[freq][intensity][repetition])
                n_trials += 1
              
        summed_traces = np.zeros(shape=(n_trials,n_samples))

        trial_counter = 0
        # let's get a sum of all our traces to average later
        for intensity in cell_traces[freq]:
            for repetition in cell_traces[freq][intensity]:
                summed_traces[trial_counter,:] = cell_traces[freq][intensity][repetition]
                trial_counter += 1

        # now we have our summed traces, so we need to average it
        avg_trace = np.average(summed_traces,axis=0)

        # now we grab the peak of the trace occuring AFTER the onset
        n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)
        response = avg_trace[n_baseline_frames:]
        area_of_response = np.trapz(response)
        tuning_curves[counter,1] = area_of_response

        counter+=1
    
    return tuning_curves

def plot_tuning_curves(cell_dictionary):
    frequency_labels = [2,4.5,10,23,52]
    intensity_labels = [30,50,70]

    # frequency_labels = [5.7,23,45] #,52]
    # intensity_labels = [0,70,80,90] #[50,70,90]

    fig,axs = plt.subplots(5,5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.5,wspace=0.001)
    axs = axs.ravel()
    counter = 0
    for cell in cell_dictionary:
        # if counter<25:
        #     counter += 1
        #     continue

        cell_tuning = cell_dictionary[cell]['tuning_curve_peak']
        smooth_cell_tuning = gaussian_filter(cell_tuning,1)
        counter += 25
        im = axs[counter-25].imshow(np.transpose(cell_tuning),cmap='winter',origin='lower')
        plt.colorbar(im,ax=axs[counter-25])
        # plt.clim(0,100)
        # axs[counter-25].set_xticks([0,2,4,6,8])
        # axs[counter-25].set_xticklabels(frequency_labels)
        # axs[counter-25].set_yticks([0,2,4])
        # axs[counter-25].set_yticklabels(intensity_labels)
        # axs[counter-25].set_xticks([0,1,2])
        # axs[counter-25].set_xticklabels(frequency_labels)
        # axs[counter-25].set_yticks([0,1,2,3])
        # axs[counter-25].set_yticklabels(intensity_labels)
        axs[counter-25].title.set_text(cell)

        counter -= 25
        if counter==24:
            break
        counter += 1

    plt.show()

def plot_single_tuning_curve(cell_dictionary,cell_IDX):

    fig = plt.figure(1)
    ax = fig.gca()

    frequency_labels = [5.7,23,45] #,52]
    intensity_labels = [0,70,80,90] #[50,70,90]

    # get cell ID at this index so we can pull its tuning curve
    cell_IDs = list(cell_dictionary.keys())
    cell_of_interest_ID = cell_IDs[cell_IDX]

    cell_tuning = cell_dictionary[cell_of_interest_ID]['tuning_curve_peak']

    im = plt.imshow(np.transpose(cell_tuning),cmap='jet',origin='lower')
    plt.colorbar(im)
    plt.xticks([0,1,2])
    ax.set_xticklabels(frequency_labels)
    plt.yticks([0,1,2,3])
    ax.set_yticklabels(intensity_labels)
    plt.show()


def get_tuning_curves(cell_dictionary):
    # for each cell, we want to add a key so our dictionary now looks like this
    # cell { traces { .... }
    #        active = T/F
    #        tuning_curve = [x,x,x,x,x,...] 
    # }
    # where tuning curve holds the average peak activity for that specific frequency 

    counter = 0
    for cell in cell_dictionary:
        if counter == CELL_OF_INTEREST:
            cell_dictionary[cell]['tuning_curve_peak'] = get_cell_tuning_by_zscore(cell_dictionary[cell]['traces'],True)
            print(cell)
        else:
            cell_dictionary[cell]['tuning_curve_peak'] = get_cell_tuning_by_zscore(cell_dictionary[cell]['traces'],False)
        
        counter+=1

    return cell_dictionary


def main():

    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)
    
    active_cell_dictionary = get_active_cells(cell_dictionary)
    cell_dictionary_with_tuning = get_tuning_curves(cell_dictionary)
    active_cell_dictionary_with_tuning = get_tuning_curves(active_cell_dictionary)

    plot_tuning_curves(active_cell_dictionary_with_tuning)
    # plot_single_tuning_curve(active_cell_dictionary,CELL_OF_INTEREST)


    with open(BASE_PATH+cell_dictionary_file_out,'wb') as f:
        pickle.dump(cell_dictionary_with_tuning,f)



if __name__ == '__main__':
    main()