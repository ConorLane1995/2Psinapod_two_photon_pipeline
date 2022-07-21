import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.lines import Line2D
from scipy.io import loadmat
import seaborn as sns

# more info: https://pietromarchesi.net/pca-neural-data.html

def reformat_epoched_data(data):
    # taking in nNeurons x nTrials x nFrames array
    # passing out nNeurons x (nTrials*nFrames) array

    n_by_f_data = np.reshape(data,(len(data),len(data[0])*len(data[0][0])))
    return n_by_f_data

def standardize(data):
    # data: n_features x n_samples

    ss = StandardScaler(with_mean=True,with_std=True)
    data_c = ss.fit_transform(data.T).T
    return data_c

def re_epoch_data(data,n_trials):
    nxtxf_data = np.reshape(data,(len(data),n_trials,-1))
    return nxtxf_data


shade_alpha      = 0.2
lines_alpha      = 0.8
pal = sns.color_palette('rocket', 13)


def add_stim_to_plot(ax):
    start_stim = 4
    end_stim = 14
    ax.axvspan(start_stim, end_stim, alpha=shade_alpha,
               color='gray')
    ax.axvline(start_stim, alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(end_stim, alpha=lines_alpha, color='gray', ls='--')
    
def add_orientation_legend(ax,trial_types):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(trial_types))]
    labels = ['{}'.format(t) for t in trial_types]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])



def main():
    data = np.load("/media/vtarka/USB DISK/Lab/2P/epoched_traces217.npy")
    conditions_mat = loadmat("/media/vtarka/USB DISK/Lab/2P/ID112_050722_PrePsilo_1.mat") # conditition type of each trial in chronological order (row 1 = trial 1)
    conditions = conditions_mat["stim_data"]

    # reformat from n x t x f array to n x t array (concatenate trials)
    n_by_f_data = reformat_epoched_data(data)

    # standardize the data so no single neuron drives the PCA purely due to 
    # its dF/F magnitude change
    data_s = standardize(n_by_f_data)

    epoched_data = re_epoch_data(data_s,len(data[0]))

    # epoched_data = data

    # trials a list of K Numpy arrays of shape N×T (number of neurons by number of time points).
    trials = []
    for trial_idx in range(len(epoched_data[0])):
        trials.append(epoched_data[:,trial_idx,:])

    trial_type = []
    for row in conditions:
        trial_type.append(row[0])

    trial_types = np.unique(conditions[:,0])
    trial_size   = trials[0].shape[1]

    t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]
   
    # prepare trial averages
    trial_averages = []
    for ind in t_type_ind:
        trial_averages.append(np.array(trials)[ind].mean(axis=0))
    Xa = np.hstack(trial_averages)

    # standardize and apply PCA
    Xa = standardize(Xa) 
    pca = PCA(n_components=3)
    Xa_p = pca.fit_transform(Xa.T).T

    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 1
    component_z = 2

    # create a boolean mask so we can plot activity during stimulus as 
    # solid line, and pre and post stimulus as a dashed line
    stim_mask = ~np.logical_and(np.arange(trial_size) >= 5,
                np.arange(trial_size) < (trial_size+1))

    # utility function to clean up and label the axes
    def style_3d_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    sigma = 2 # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views
    fig = plt.figure(figsize=[9, 4])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    for ax in axs:
        for t, t_type in enumerate(trial_types):

            # for every trial type, select the part of the component
            # which corresponds to that trial type:
            x = Xa_p[component_x, t * trial_size :(t+1) * trial_size]
            y = Xa_p[component_y, t * trial_size :(t+1) * trial_size]
            z = Xa_p[component_z, t * trial_size :(t+1) * trial_size]
            
            # apply some smoothing to the trajectories
            x = gaussian_filter1d(x, sigma=sigma)
            y = gaussian_filter1d(y, sigma=sigma)
            z = gaussian_filter1d(z, sigma=sigma)

            # use the mask to plot stimulus and pre/post stimulus separately
            z_stim = z.copy()
            z_stim[stim_mask] = np.nan
            z_stim[4] = z[4]
            z_prepost = z.copy()
            z_prepost[~stim_mask] = np.nan

            ax.plot(x, y, z_stim, c = pal[t])
            ax.plot(x, y, z_prepost, c=pal[t], ls=':')
            
            # plot dots at initial point
            ax.scatter(x[0], y[0], z[0], c=pal[t], s=14)
            
            # make the axes a bit cleaner
            style_3d_ax(ax)
            
    # specify the orientation of the 3d plot        
    ax1.view_init(elev=22, azim=30)
    ax2.view_init(elev=22, azim=110)
    plt.tight_layout()
    add_orientation_legend(ax1,trial_types)
    plt.show()


if __name__=="__main__":
    main()