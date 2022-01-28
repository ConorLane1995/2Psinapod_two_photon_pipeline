import matplotlib as mpl
import numpy as np
import pickle


BASE_PATH = "D:/vid127_pseudorandom_stim/"
stim_path = "Stim_Data_PseudoRandom_vid127.npy"
epoched_rec_path = "epoched_F.npy"
output_file = "cell_traces_with_stims.pkl"

def format_trials(traces,stim):

    # traces should be an nTrial x nFrame array
    # stim should be an nTrial x 4 array (info on this structure in the README.md)

    # need this to return a dictionary that will be contained within this cell key in the big dict

    # format the dictionary so we get this structure:
    # cell_n{ 
    #     freq_f{
    #           intensity_i{
    #                   trace = [x,x,x,x,...]
    #                       }
    #            }
    # }

    # use the frequencies we played as the keys of our dictionary (outermost dictionary)
    freq_dict = dict.fromkeys(np.unique(stim[:,0]))

    # nest our intensities inside our freq dictionary
    for freq in freq_dict:
        freq_dict[freq] = dict.fromkeys(np.unique(stim[:,1]))

    # make empty dictionaries so we can index properly later
    for freq in freq_dict:
        # print(type(freq))
        for intensity in freq_dict[freq]:
            freq_dict[freq][intensity] = {}

    # make a really shitty temporary map so we can keep track of how many repetitions of this trial we've seen
    # just going to add together the frequency and intensity to index it
    # biggest element we'll need is max(frequency) + max(intensity)
    max_element = max(stim[:,0]) + max(stim[:,1]) + 10
    temp_map = [0] * max_element

    for trial in range(len(stim)):

        # trial's frequency
        f = stim[trial,0]

        # trial's intensity
        i = stim[trial,1]

        num_rep = temp_map[f+i]+1
        temp_map[f+i] += 1

        # using the frequency and intensity to index our dictionary to store our trace
        freq_dict[f][i][num_rep] = traces[trial,:]

    return freq_dict

def format_all_cells(epoched_traces,stim):

    # make a dictionary where each cell is one key
    num_cells = len(epoched_traces)
    d = dict.fromkeys(range(1,num_cells+1))

    # for each cell
    # format the dictionary so we get this structure:
    # cell_n{ 
    #     freq_f{
    #           intensity_i{
    #                   trace = [x,x,x,x,...]
    #                       }
    #            }
    # }

    for cell in d:
        d[cell] = format_trials(epoched_traces[cell-1,:,:],stim)
    
    return d

def main():
    stim = np.load(BASE_PATH + stim_path,allow_pickle=True)
    traces = np.load(BASE_PATH + epoched_rec_path,allow_pickle=True)

    mega_dict = format_all_cells(traces,stim)

    with open(BASE_PATH+output_file,'wb') as f:
        pickle.dump(mega_dict,f)

if __name__=="__main__":
    main()