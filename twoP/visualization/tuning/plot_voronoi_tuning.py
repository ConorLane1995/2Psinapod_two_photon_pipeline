from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
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
def get_voronoi_polygons(coordinates):
    vor = Voronoi(coordinates)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    return polygons

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
    pal = sns.color_palette('rocket_r', 13)
    colors = []
    for bf in BFs:
        colors.append(pal[bf])

    return colors

def plot_polygons(polygons, colors, ax=None, alpha=0.5, linewidth=1, saveas=None, show=True):
    # Configure plot 
    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis("equal")

    # Set limits
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    # Add polygons 
    for poly,color in zip(polygons,colors):
        colored_cell = Polygon(poly,
                               linewidth=linewidth, 
                               alpha=alpha,
                               facecolor=color,
                               edgecolor="black")
        ax.add_patch(colored_cell)

    if not saveas is None:
        plt.savefig(saveas)
    if show:
        plt.show()

    return ax 

def main():
    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    with open(BASE_PATH + "recording_info.pkl", 'rb') as f:
        recording_info = pickle.load(f)

    active_cell_dict = get_active_cells(cell_dict)
    coordinates = get_coordinates(active_cell_dict)

    BFs = []
    for cell in active_cell_dict:
        BFs.append(get_best_frequency(active_cell_dict[cell]['tuning'],recording_info['frequencies']))

    colors = get_BF_colors(BFs)
    polygons = get_voronoi_polygons(coordinates)

    v = Voronoi(coordinates)
    voronoi_plot_2d(v)
    plt.show()

    # plot_polygons(polygons,colors)

if __name__=="__main__":
    main()

#https://stackoverflow.com/questions/63347564/how-do-you-select-custom-colours-to-fill-regions-of-a-voronoi-diagram-using-scip