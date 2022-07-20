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
    data = np.load("/media/vtarka/USB DISK/Lab/2P/active_epoched_traces217.npy")
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

    Xl = np.hstack(trials)


    Xl = standardize(Xl)
    pca = PCA(n_components=15)
    Xl_p = pca.fit_transform(Xl.T).T


    n_components=15

    gt = {comp : {t_type : [] for t_type in trial_types} for comp in range(n_components)}

    for comp in range(n_components):
        for i, t_type in enumerate(trial_type):
            t = Xl_p[comp, trial_size * i: trial_size * (i + 1)]
            gt[comp][t_type].append(t)

    for comp in range(n_components):
        for t_type in trial_types:
            gt[comp][t_type] = np.vstack(gt[comp][t_type])


    f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
    for comp in range(3):
        ax = axes[comp]
        for t, t_type in enumerate(trial_types):
            tmp = gt[comp][t_type]
            sns.lineplot(y=np.mean(gt[comp][t_type],axis=0), x=range(15), ax=ax,
                    err_style='band',
                    ci=95,
                    color=pal[t])
        add_stim_to_plot(ax)
        ax.set_ylabel('PC {}'.format(comp+1))
    axes[1].set_xlabel('Time (s)')
    sns.despine(right=True, top=True)
    add_orientation_legend(axes[2],trial_types)
    plt.show()





if __name__=="__main__":
    main()