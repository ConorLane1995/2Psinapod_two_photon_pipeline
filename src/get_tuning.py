import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("../")
from utils import get_active_cells
from scipy.stats import zscore

BASE_PATH = "D:/Vid_139/"
cell_dictionary_file = "cells3.pkl"
cell_dictionary_file_out = "cells4.pkl"
EPOCH_START_IN_MS = -500 # time before trial onset included in the epoch
EPOCH_END_IN_MS = 2500 # time after trial onset included in the epoch
FRAMERATE = 10

# def get_cell_tuning_by_peak(cell_traces):

    
    # # cell_traces is a dictionary of frequencies 
    # # under each frequency is a dictionary of intensities
    # # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # # allocate some space to return
    # # we want a matrix that is nFrequencies x 2 (first column is the frequency, second column is the activity)
    # tuning_curves = np.empty((len(cell_traces),2))

    # counter = 0 # to keep track of where we're indexing the empty array
    # for freq in cell_traces:
    #     tuning_curves[counter,0] = freq
    #     # print(tuning_curves[counter,0])

    #     # now we find the number of intensities we presented at
    #     n_intensities = len(cell_traces[freq].keys())

    #     # now we make a temporary vector to append to the tuning curve at the end of this loop
    #     activation_per_intensity = np.empty((n_intensities,1))

    #     # 
    #     # now we need to get the peak intensity of the average of all the trials for that one frequency 
    #     n_samples = 0
    #     n_trials = 0
    #     for intensity in cell_traces[freq]:
    #         for repetition in cell_traces[freq][intensity]:
    #             if n_trials == 0:
    #                 n_samples = len(cell_traces[freq][intensity][repetition])
    #             n_trials += 1
              
    #     summed_traces = np.zeros(shape=(n_trials,n_samples))

    #     trial_counter = 0
    #     # let's get a sum of all our traces to average later
    #     for intensity in cell_traces[freq]:
    #         for repetition in cell_traces[freq][intensity]:
    #             summed_traces[trial_counter,:] = cell_traces[freq][intensity][repetition]
    #             trial_counter += 1

    #     # now we have our summed traces, so we need to average it
    #     avg_trace = np.average(summed_traces,axis=0)

    #     # now we grab the peak of the trace occuring AFTER the onset
    #     n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)
    #     response = avg_trace[n_baseline_frames:]
    #     peak_response = np.amax(response)
    #     tuning_curves[counter,1] = peak_response

    #     counter+=1
    
    # return tuning_curves

def get_cell_tuning_by_peak(cell_traces):
    # cell_traces is a dictionary of frequencies 
    # under each frequency is a dictionary of intensities
    # under each intensity are the traces for each repetitiong of that frequency/intensity combination

    # allocate some space to return
    # we want a matrix that is nFrequencies x nIntensities 
    tuning_curves = np.empty((len(cell_traces),len(cell_traces[next(iter(cell_traces))].keys())))

    frequency_counter = 0 # to keep track of where we're indexing the empty array
    for freq in cell_traces:
        intensity_counter = 0

        # find the number of intensities we presented at
        n_intensities = len(cell_traces[freq].keys())

        # make a temporary vector to append to the tuning curve at the end of this loop
        # we will fill one n_intensities length column of the 2D matrix we are returning
        activation_per_intensity = np.empty((n_intensities,1))

        # iterate through each intensity the frequency was presented at
        for intensity in cell_traces[freq]:
            # collect all the trials of this one frequency presented at this one intensity
            # it will be an nTrials x nFrames matrix
            all_trials_of_this_intensity = []
            n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1
            # iterate through each trial of this frequency/intensity combination
            counter=0
            for trial in cell_traces[freq][intensity]:
                # plt.plot(cell_traces[freq][intensity][trial][n_baseline_frames:])
                counter+=1
                all_trials_of_this_intensity.append(cell_traces[freq][intensity][trial])

            # plt.show()

            # convert the matrix of trials into a np array
            all_trials_as_np = np.array(all_trials_of_this_intensity)

            # average across all the trials to get a 1 x nFrames vector
            average_trial_of_this_intensity = np.average(all_trials_as_np, axis=0)

            # now we grab the peak of the trace occuring AFTER the onset
            n_baseline_frames = round(EPOCH_START_IN_MS/1000 * FRAMERATE)*-1
            baseline = average_trial_of_this_intensity[0:n_baseline_frames]
            baseline_mean = np.average(baseline)
            baseline_std = np.std(baseline)

            zscorer = lambda x: (x-baseline_mean)/baseline_std

            response = average_trial_of_this_intensity[n_baseline_frames:]
            zscore_response = np.array([zscorer(xi) for xi in response])

            # response = average_trial_of_this_intensity[n_baseline_frames:]
            # zscore_response = zscore(response)
            # plt.plot(response)
            peak_response = np.amax(zscore_response)
            # peak_response = np.amax(response)
            # peak_response = np.trapz(response)
            tuning_curves[frequency_counter,intensity_counter] = peak_response

            intensity_counter += 1

        frequency_counter += 1
        # plt.show()
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
    fig,axs = plt.subplots(5,5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.5,wspace=0.001)
    axs = axs.ravel()
    counter = 0
    for cell in cell_dictionary:
        if counter<25:
            counter += 1
            continue

        cell_tuning = cell_dictionary[cell]['tuning_curve']

        im = axs[counter-25].imshow(np.transpose(cell_tuning),cmap='hot',origin='lower')
        plt.colorbar(im,ax=axs[counter-25])

        if counter==49:
            break
        counter += 1

    plt.show()

def get_tuning_curves(cell_dictionary):
    # for each cell, we want to add a key so our dictionary now looks like this
    # cell { traces { .... }
    #        active = T/F
    #        tuning_curve = [x,x,x,x,x,...] 
    # }
    # where tuning curve holds the average peak activity for that specific frequency 

    for cell in cell_dictionary:
        cell_dictionary[cell]['tuning_curve'] = get_cell_tuning_by_peak(cell_dictionary[cell]['traces'])

    return cell_dictionary


def main():

    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)
    
    active_cell_dictionary = get_active_cells(cell_dictionary)
    cell_dictionary_with_tuning = get_tuning_curves(active_cell_dictionary)

    plot_tuning_curves(active_cell_dictionary)


    with open(BASE_PATH+cell_dictionary_file_out,'wb') as f:
        pickle.dump(cell_dictionary_with_tuning,f)



if __name__ == '__main__':
    main()