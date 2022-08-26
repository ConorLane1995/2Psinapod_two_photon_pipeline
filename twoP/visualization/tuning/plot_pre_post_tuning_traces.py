"""
Plots a side by side of pre and post tuning for the cells specified in CELLS_TO_PLOT
INPUT: cell dictionaries for the pre recording and post recording, recording_info.pkl for one of the two recordings (doesn't matter which)
OUTPUT: (optional) saved figures for every cell listed in CELLS_TO_PLOT
AUTHOR: Veronica Tarka, August 2022, veronica.tarka@mail.mcgill.ca
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

PRE_CELL_DICT_FILE = "/media/vtarka/USB DISK/Lab/2P/Vid_246/cells.pkl" # path to the cell dictionary for the pre-recording
POST_CELL_DICT_FILE = "/media/vtarka/USB DISK/Lab/2P/Vid_247/cells.pkl" # path to the cell dictionary for the post-recording
RECORDING_INFO_FILE = "/media/vtarka/USB DISK/Lab/2P/Vid_246/recording_info.pkl" # path to the recording info file, can be for just one of the recordings since they should use the same stim protocol
FIGURE_FOLDER = None # folder to save the figures in, set to None if you don't want them saved

# set your list of cell IDs you'd like to plot
active_cells = np.load("/media/vtarka/USB DISK/Lab/2P/Vid_247/active_cells.npy")
CELLS_TO_PLOT = random.sample(list(active_cells),10) #[2,5,10,27,31,39,48,52,81,103,123,133,150,240,257,1475]

"""
Get the largest value of the recording
@param cell_traces: the contents of 'traces' for a single cell in the big dictionary
@return the maximum value across all of the trial traces
"""
def get_max_response(cell_traces):

    whole_trace = []
    for freq in cell_traces:
        for intensity in cell_traces[freq]:
            for rep in cell_traces[freq][intensity]:
                whole_trace.append(cell_traces[freq][intensity][rep])
    
    return np.max(whole_trace)
                
"""
Plot the tuning traces for a single cell
@param cell_pre: the contents of a single cell key in the big dictionary (the pre-psilo recording)
@param cell_post: the contents of the same single cell key in the big dictionary (the post-psilo recording)
@param cell: the integer ID number of the cell (the key used to access the cell in the big dictionary)
@param frequencies: a list of the unique frequencies presented during the recording (in ascending order)
@param intensities: a list of the unique intensities presented during the recording (in ascending order)
@param y_limit: how tall the y axis should be for each subplot
"""
def plot_tuning_traces(cell_pre,cell_post,cell,frequencies,intensities,y_limit):

    f = plt.figure(figsize=(26,10))
    outer = gridspec.GridSpec(2,2,height_ratios=[4,1])

    # get number of frequencies and intensities
    n_frequencies = len(frequencies)
    n_intensities = len(intensities)

    # PRE first
    # add some labeling first
    ax = f.add_subplot(outer[0])
    if cell_pre['active']:
        plt.title("Pre (active)")
    else:
        plt.title("Pre (not active)")
    ax.axis('off')

    # make our tuning curve grid
    inner = gridspec.GridSpecFromSubplotSpec(n_intensities,n_frequencies,subplot_spec=outer[0],wspace=0,hspace=0)
    cell_traces_pre = cell_pre['traces']
    for row,freq in zip(range(n_frequencies),cell_traces_pre.keys()):
        for col,itsy in zip(range(n_intensities),reversed(list(cell_traces_pre[freq].keys()))):

            ax = plt.Subplot(f, inner[col,row]) # make a subplot

            for rep in cell_traces_pre[freq][itsy]:
                ax.plot(cell_traces_pre[freq][itsy][rep]) # plot every trial

            # miscellaneous formatting
            ax.set_xticks([])
            ax.set_yticks([])
            if row==0:
                ax.set_ylabel(itsy) # add the intensity to the far left edge
            if col==n_intensities-1:
                ax.set_xlabel(freq) # add the frequency at the bottom
            ax.axvline(x=4,color='k',linestyle='--',lw=0.2)
            ax.set_ylim(bottom=0,top=y_limit)
            ax.autoscale(enable=True, axis='x', tight=True)

            f.add_subplot(ax)

    # now POST
    # add titles first
    ax = f.add_subplot(outer[1])
    if cell_post['active']:
        plt.title("Post (active)")
    else:
        plt.title("Post (not active)")
    ax.axis('off')

    inner = gridspec.GridSpecFromSubplotSpec(n_intensities,n_frequencies,subplot_spec=outer[1],wspace=0,hspace=0)
    cell_traces_post = cell_post['traces']
    for row,freq in zip(range(n_frequencies),cell_traces_post.keys()):
        for col,itsy in zip(range(n_intensities),reversed(list(cell_traces_post[freq].keys()))):

            ax = plt.Subplot(f, inner[col,row])

            for rep in cell_traces_post[freq][itsy]:
                ax.plot(cell_traces_post[freq][itsy][rep]) # plot every trial

            # miscellaneous formatting
            ax.set_xticks([])
            ax.set_yticks([])
            if row==0:
                ax.set_ylabel(itsy) # add the intensity to the far left edge
            if col==n_intensities-1:
                ax.set_xlabel(freq) # add the frequency at the bottom
            ax.axvline(x=4,color='k',linestyle='--',lw=0.2)
            ax.set_ylim(bottom=0,top=y_limit)
            ax.autoscale(enable=True, axis='x', tight=True)

            f.add_subplot(ax)

    # add the tuning heat maps
    # PRE
    ax = f.add_subplot(outer[2])
    im = ax.imshow(np.transpose(cell_pre['tuning']),cmap='winter',origin='lower')
    ax.set_xticks(range(0,n_frequencies,2))
    ax.set_xticklabels(frequencies[range(0,n_frequencies,2)])
    ax.set_yticks(range(0,len(intensities)))
    ax.set_yticklabels(intensities)
    plt.colorbar(im,ax=ax,format=lambda x, _:f"{x:4.0f}")

    # POST
    ax = f.add_subplot(outer[3])
    im = ax.imshow(np.transpose(cell_post['tuning']),cmap='winter',origin='lower')
    ax.set_xticks(range(0,n_frequencies,2))
    ax.set_xticklabels(frequencies[range(0,n_frequencies,2)])
    ax.set_yticks(range(0,len(intensities)))
    ax.set_yticklabels(intensities)
    plt.colorbar(im,ax=ax,format=lambda x, _:f"{x:4.0f}")

    f.text(0.5,0.01,"Frequency (Hz)",va='center',ha='center')
    f.text(0.01,0.5,"Intensity (dB)",va='center',ha='center',rotation='vertical')
    f.suptitle(cell)
    if FIGURE_FOLDER != None:
        plt.savefig(FIGURE_FOLDER+"pre_post_{}.png".format(cell))
    plt.show(block=False)


def main():

    with open(PRE_CELL_DICT_FILE,"rb") as f:
        r1_cells = pickle.load(f)

    with open(POST_CELL_DICT_FILE,"rb") as f:
        r2_cells = pickle.load(f)

    # load the recording info file
    with open(RECORDING_INFO_FILE,"rb") as f:
        recording_info = pickle.load(f)

    frequencies = recording_info['frequencies']
    intensities = recording_info['intensities']

    for cell in CELLS_TO_PLOT:
        highest_point = max([get_max_response(r1_cells[cell]['traces']),get_max_response(r2_cells[cell]['traces'])])
        plot_tuning_traces(r1_cells[cell],r2_cells[cell],cell,frequencies,intensities,highest_point+(highest_point*0.1))
    
    plt.show()
    


if __name__=="__main__":
    main()