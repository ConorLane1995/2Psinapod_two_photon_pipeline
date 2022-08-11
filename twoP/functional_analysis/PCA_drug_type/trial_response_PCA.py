import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
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
pal = sns.color_palette('husl', 2)


# def add_stim_to_plot(ax):
#     ax.axvspan(start_stim, end_stim, alpha=shade_alpha,
#                color='gray')
#     ax.axvline(start_stim, alpha=lines_alpha, color='gray', ls='--')
#     ax.axvline(end_stim, alpha=lines_alpha, color='gray', ls='--')
    
def add_orientation_legend(ax,trial_types):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(trial_types))]
    labels = ['{}'.format(t) for t in trial_types]
    labels = ['Pre-Psilocybin','Post-Psilocybin']
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.9,1])



def main():
    data_pre = np.load("/media/vtarka/USB DISK/Lab/2P/epoched_traces217.npy")
    data_post = np.load("/media/vtarka/USB DISK/Lab/2P/epoched_traces220.npy")

    # reformat from n x t x f array to n x t array (concatenate trials)
    n_by_f_data_pre = reformat_epoched_data(data_pre)
    n_by_f_data_post = reformat_epoched_data(data_post)

    n_by_f_data = np.concatenate((n_by_f_data_post,n_by_f_data_pre))

    # standardize the data so no single neuron drives the PCA purely due to 
    # its dF/F magnitude change
    data_s = standardize(n_by_f_data)

    epoched_data = re_epoch_data(data_s,len(data_pre[0]))

    # epoched_data = data

    # trials a list of K Numpy arrays of shape N×T (number of neurons by number of time points).
    trials = []
    for neur_idx in range(len(epoched_data)):
        trials.append(epoched_data[neur_idx,:,:])

    trial_type = np.concatenate((np.zeros(len(data_pre)),np.ones(len(data_post))))
    trial_types = np.unique(trial_type)

    Xr = np.vstack([t[:,:].mean(axis=1) for t in trials]).T
    Xr_sc = standardize(Xr)

    pca = PCA(n_components=15)
    Xp = pca.fit_transform(Xr_sc.T).T
    

    total_points = 0
    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        for t, t_type in enumerate(trial_types):
            tmp = np.nonzero(trial_type==t_type)
            x = Xp[proj[0], np.nonzero(trial_type==t_type)]
            y = Xp[proj[1], np.nonzero(trial_type==t_type)]
            print(len(x[0]))
            total_points += len(x[0])
            ax.scatter(x, y, c=pal[t], s=30, alpha=0.7)
            ax.set_xlabel('PC {}'.format(proj[0]+1))
            ax.set_ylabel('PC {}'.format(proj[1]+1))

        print(total_points)
        total_points = 0
    sns.despine(fig=fig, top=True, right=True)

    add_orientation_legend(axes[2],trial_types)
    print(pca.explained_variance_ratio_)
    plt.show()


if __name__=="__main__":
    main()