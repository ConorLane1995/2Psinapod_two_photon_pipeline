import numpy as np
import pickle

BASE_PATH = "D:/vid127_pseudorandom_stim/"
cell_dictionary_file = "traces_with_activity_boolean_2.pkl"
cell_dictionary_file_out = "traces_with_activity_boolean_3.pkl"
EPOCH_START_IN_MS = -500 # time before trial onset included in the epoch
EPOCH_END_IN_MS = 2700 # time after trial onset included in the epoch
FRAMERATE = 10

def get_cell_tuning(cell_traces):
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
        peak_response = np.amax(response)
        tuning_curves[counter,1] = peak_response

        counter+=1
    
    return tuning_curves


def get_tuning_curves(cell_dictionary):
    # for each cell, we want to add a key so our dictionary now looks like this
    # cell { traces { .... }
    #        active = T/F
    #        tuning_curve = [x,x,x,x,x,...] 
    # }
    # where tuning curve holds the average peak activity for that specific frequency 

    for cell in cell_dictionary:
        cell_dictionary[cell]['tuning_curve'] = get_cell_tuning(cell_dictionary[cell]['traces'])

    return cell_dictionary


def main():

    with open(BASE_PATH + cell_dictionary_file, 'rb') as f:
        cell_dictionary = pickle.load(f)
    
    cell_dictionary_with_tuning = get_tuning_curves(cell_dictionary)

    with open(BASE_PATH+cell_dictionary_file_out,'wb') as f:
        pickle.dump(cell_dictionary_with_tuning,f)



if __name__ == '__main__':
    main()