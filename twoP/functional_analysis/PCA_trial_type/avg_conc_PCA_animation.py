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

    t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]
   
    # fit PCA on trial averages
    trial_averages = []
    for ind in t_type_ind:
        trial_averages.append(np.array(trials)[ind].mean(axis=0))
    Xav = np.hstack(trial_averages)

    ss = StandardScaler(with_mean=True, with_std=True)
    Xav_sc = ss.fit_transform(Xav.T).T
    pca = PCA(n_components=15) 
    pca.fit(Xav_sc.T) # only call the fit method


    n_components = 15

    projected_trials = []
    for trial in trials:
        # scale every trial using the same scaling applied to the averages 
        trial = ss.transform(trial.T).T
        # project every trial using the pca fit on averages
        proj_trial = pca.transform(trial.T).T
        projected_trials.append(proj_trial)

    gt = {comp: {t_type: [] for t_type in trial_types}
      for comp in range(n_components)}

    for comp in range(n_components):
        for i, t_type in enumerate(trial_types):
            t = projected_trials[i][comp, :]
            gt[comp][t_type].append(t)


    f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True)
    for comp in range(3):
        ax = axes[comp]
        for t, t_type in enumerate(trial_types):
            sns.lineplot(y=np.average(gt[comp][t_type],axis=0), x=range(15), ax=ax,
                    err_style='band',
                    ci=95,
                    color=pal[t])
        add_stim_to_plot(ax)
        ax.set_ylabel('PC {}'.format(comp+1))
    axes[1].set_xlabel('Time (s)')
    sns.despine(right=True, top=True)
    add_orientation_legend(axes[2],trial_types)
    # plt.show()

    # import the animation and the HTML module to create and render the animation
    from matplotlib import animation 
    from IPython.display import HTML

    time = range(15)

    # smooth the single projected trials 
    for i in range(len(projected_trials)):
        for c in range(projected_trials[0].shape[0]):
            projected_trials[i][c, :] = projected_trials[i][c, :]#gaussian_filter1d(projected_trials[i][c, :], sigma=1)

    # for every time point (imaging frame) get the position in PCA space of every trial
    pca_frame = []
    for t in range(trial_size):
        # projected data for all trials at time t 
        Xp = np.hstack([tr[:, None, t] for tr in projected_trials]).T
        pca_frame.append(Xp)
        
    subspace = (0,2) # pick the subspace given by the second and third components
        
    # set up the figure
    fig, ax = plt.subplots(1, 1, figsize=[6, 6]); plt.close()
    ax.set_xlim(( -16, 16))
    ax.set_ylim((-16, 16))
    ax.set_xlabel('PC 2')
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_ylabel('PC 3')
    sns.despine(fig=fig, top=True, right=True)

    # generate empty scatter plot to be filled by data at every time point
    scatters = []
    for t, t_type in enumerate(trial_types):
        scatter, = ax.plot([], [], 'o', lw=2, color=pal[t])
        scatters.append(scatter)

    # red dot to indicate when stimulus is being presented
    stimdot, = ax.plot([], [], 'o', c='r', markersize=35, alpha=0.5)

    # annotate with stimulus and time information
    text     = ax.text(6.3, 9, 'Stimulus OFF \nt = {:.2f}'.format(time[0]), fontdict={'fontsize':14})

    # this is the function to be called at every animation frame
    def animate(i):
        for t, t_type in enumerate(trial_types):
            # find the x and y position of all trials of a given type
            x = pca_frame[i][t_type_ind[t], subspace[0]]
            y = pca_frame[i][t_type_ind[t], subspace[1]]
            # update the scatter
            scatters[t].set_data(x, y)
            
        # update stimulus and time annotation
        if (i > 4) and (i < (trial_size+1)):
            stimdot.set_data(10, 14)
            text.set_text('Stimulus ON \nt = {:.2f}'.format(time[i]))
        else:
            stimdot.set_data([], [])
            text.set_text('Stimulus OFF \nt = {:.2f}'.format(time[i]))
        return (scatter,)

    # generate the animation
    anim = animation.FuncAnimation(fig, animate, 
                                frames=len(pca_frame), interval=1000,
                                blit=True)

    anim.save('PCA.mp4',fps=1) #, extra_args=['-vcodec','libx264'])
    plt.show()

    HTML(anim.to_jshtml()) # render animation
    plt.show()







if __name__=="__main__":
    main()