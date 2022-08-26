"""
TODO doc
"""

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pickle
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../../')
from utils import get_active_cells

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']


def get_coordinates(cell_dict):
    coordinates = []

    for cell in cell_dict:
        coordinates.append([cell_dict[cell]['x'],cell_dict[cell]['y']])

    return np.array(coordinates)

def get_best_frequency(cell_tuning,freqs):
    median_across_itsies = np.median(cell_tuning, axis=1)
    max_response_idx = np.argmax(median_across_itsies)
    return max_response_idx

def get_BF_colors(BFs):
    pal = sns.color_palette('rocket_r', 12)
    colors = []
    for bf in BFs:
        colors.append(pal[bf])

    return colors

def main():
    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    with open(BASE_PATH + "recording_info.pkl", 'rb') as f:
        recording_info = pickle.load(f)

    freqs = recording_info['frequencies']
    active_cell_dict = get_active_cells(cell_dict)
    coordinates = get_coordinates(active_cell_dict)
    coordinates = np.append(coordinates, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)

    BFs = []
    for cell in active_cell_dict:
        BFs.append(get_best_frequency(active_cell_dict[cell]['tuning'],recording_info['frequencies']))

    v = Voronoi(coordinates)
    voronoi_plot_2d(v,show_vertices=False,show_points=False)

    pal = sns.color_palette('rocket', 12)
    for j,bf in zip(range(len(coordinates)),BFs):
        region = v.regions[v.point_region[j]]
        if not -1 in region:
            polygon = [v.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=pal[bf])

    
    # plt.plot(coordinates[:,0], coordinates[:,1], 'ko')
    plt.xlim([coordinates[:-4,:].min() - 20, coordinates[:-4,:].max() + 20])
    plt.ylim([coordinates[:-4,:].min() - 20, coordinates[:-4,:].max() + 20])

    # add a color bar showing the range of values we're looking at
    my_cmap = ListedColormap(sns.color_palette('rocket',12).as_hex())
    cbar = plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,vmax=11),cmap=my_cmap),ticks=range(0,len(freqs),2))
    cbar.ax.set_yticklabels([freqs[f] for f in range(0,len(freqs),2)])
    plt.show()


if __name__=="__main__":
    main()

#https://stackoverflow.com/questions/63347564/how-do-you-select-custom-colours-to-fill-regions-of-a-voronoi-diagram-using-scip