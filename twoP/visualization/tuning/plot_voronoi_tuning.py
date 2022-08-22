"""
TODO doc
"""

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
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



def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

"""
https://www.daniweb.com/programming/computer-science/tutorials/520314/how-to-make-quality-voronoi-diagrams
"""

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

    colors = get_BF_colors(BFs)
    for i in range(4):
        colors.append((0,0,0))

    # for color,counter in zip(colors,range(len(colors))):
    #     plt.plot(range(5),np.ones(5)*counter,c=color)

    # plt.show()

    # np_colors = np.array(colors)
    # colors = np.multiply(np_colors,100)
    # colors = np.round(colors)
    # colors = colors.astype(int)

    # colours = list(lambda flag: '#%02X%02X%02X' % (c[i,0],c[i,1],c[i,2]) for i in range(len(c)))


    v = Voronoi(coordinates)
    # fig = plt.figure()
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