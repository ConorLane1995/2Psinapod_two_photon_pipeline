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


# def add_stim_to_plot(ax):
#     ax.axvspan(start_stim, end_stim, alpha=shade_alpha,
#                color='gray')
#     ax.axvline(start_stim, alpha=lines_alpha, color='gray', ls='--')
#     ax.axvline(end_stim, alpha=lines_alpha, color='gray', ls='--')
    
def add_orientation_legend(ax,trial_types):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(trial_types))]
    labels = ['{}'.format(t) for t in trial_types]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])



def main():
    data_pre = np.load("/media/vtarka/USB DISK/Lab/2P/active_epoched_traces217.npy")
    data_post = np.load("/media/vtarka/USB DISK/Lab/2P/active_epoched_traces220.npy")
    
    conditions_mat_pre = loadmat("/media/vtarka/USB DISK/Lab/2P/ID112_050722_PrePsilo_1.mat") # conditition type of each trial in chronological order (row 1 = trial 1)
    conditions_pre = conditions_mat_pre["stim_data"]
    pre_trials = conditions_pre[:,0]
    pre_drug_types = np.zeros(len(data_pre))

    conditions_mat_post = loadmat("/media/vtarka/USB DISK/Lab/2P/ID112_070522_PostPsilo_1.mat") # conditition type of each trial in chronological order (row 1 = trial 1)
    conditions_post = conditions_mat_post["stim_data"]
    post_trials = conditions_post[:,0]
    post_drug_types = np.ones(len(data_post))

    # sort data by the conditions so we can concatenate
    pre_trials_idx = pre_trials.argsort()
    data_pre_sorted = []
    for i,idx in enumerate(pre_trials_idx):
        data_pre_sorted.append(data_pre[:,idx,:])

    data_pre_sorted = np.transpose(data_pre_sorted,(1,0,2))

    post_trials_idx = post_trials.argsort()
    data_post_sorted = []
    for i,idx in enumerate(post_trials_idx):
        data_post_sorted.append(data_post[:,idx,:])

    data_post_sorted = np.transpose(data_post_sorted,(1,0,2))

    data = np.concatenate((data_pre_sorted,data_post_sorted))

    # reformat from n x t x f array to n x t array (concatenate trials)
    # n_by_f_data = reformat_epoched_data(data)

    # standardize the data so no single neuron drives the PCA purely due to 
    # its dF/F magnitude change
    # data_s = standardize(n_by_f_data)

    epoched_data = data #re_epoch_data(data_s,len(data[0]))
    trial_type = np.sort(pre_trials)
    drug_type = np.concatenate((pre_drug_types,post_drug_types))

    # trials a list of K Numpy arrays of shape N×T (number of neurons by number of time points).
    trials = []
    for trial_idx in range(len(epoched_data[0])):
        trials.append(epoched_data[:,trial_idx,:])

    trial_types = np.unique(trial_type)
    drug_types = np.unique(drug_type)

    Xr = np.vstack([t[:,4:].mean(axis=1) for t in trials]).T
    Xr_sc = standardize(Xr)

    pca = PCA(n_components=15)
    Xp = pca.fit_transform(Xr_sc.T).T
    
    mrk_sty = ["o","X"]
    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        for t, t_type in enumerate(trial_types):
            # for d, d_type in enumerate(drug_types):
            tmp = np.nonzero(trial_type==t_type)
            x = Xp[proj[0], np.nonzero(trial_type==t_type)]
            y = Xp[proj[1], np.nonzero(trial_type==t_type)]
            print(len(x[0]))
            ax.scatter(x, y, c=pal[t], s=30, alpha=0.7)
            ax.set_xlabel('PC {}'.format(proj[0]+1))
            ax.set_ylabel('PC {}'.format(proj[1]+1))

    sns.despine(fig=fig, top=True, right=True)

    add_orientation_legend(axes[2],trial_types)
    print(pca.explained_variance_ratio_)
    plt.show()


if __name__=="__main__":
    main()