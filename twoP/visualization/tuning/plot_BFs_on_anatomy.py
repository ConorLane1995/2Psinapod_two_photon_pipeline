"""
TODO doc
"""


import pickle
import json
import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from PIL.ImageOps import grayscale

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../../')
from utils import get_active_cells

# load what we need from the config file
with open(os.path.abspath(os.path.dirname(__file__)) +'/../../../config.json','r') as f:
    config = json.load(f)

BASE_PATH = config['RecordingFolder']
CELL_DICT_FILE = config['AnalysisFile']

def get_pixel_color(cell_tuning,freqs):
    median_across_itsies = np.median(cell_tuning, axis=1)
    max_response_idx = np.argmax(median_across_itsies)

    pal = sns.color_palette('rocket', 12,)
    color = np.array(pal[max_response_idx])
    color = np.multiply(color,100)
    color = color.astype(int)
    tint_factor = 0.2
    color[0] = color[0] + (255 - color[0]) * tint_factor
    # color[1] = color[1] + (255 - color[1]) * 0.05
    color[2] = color[2] + (255 - color[2]) * tint_factor
    return tuple(color)

def convertLToRgb(src):
    src.load()   
    band = src if Image.getmodebands(src.mode) == 1 else grayscale(src)
    return Image.merge('RGB', (band, band, band))



def main():
    # load the dictionary file
    with open(BASE_PATH + CELL_DICT_FILE, 'rb') as f:
        cell_dict = pickle.load(f)

    with open(BASE_PATH + "recording_info.pkl", 'rb') as f:
        recording_info = pickle.load(f)

    # ops = np.load(BASE_PATH + "ops.npy",allow_pickle=True)
    # im = ops['refImg']
    freqs = recording_info['frequencies']

    anat = Image.open(BASE_PATH + "AVG_file000_chan0.jpg")
    rgb_img = convertLToRgb(anat)

    filter = ImageEnhance.Color(anat)
    bw_anat = filter.enhance(0)
    bw_anat = np.array(bw_anat)
    bwa = np.zeros(shape=(512,512,3))
    bwa[:,:,0] = bw_anat
    bw_anat = np.array(rgb_img)
    active_cell_dict = get_active_cells(cell_dict)
    for cell in active_cell_dict:

        color = get_pixel_color(active_cell_dict[cell]['tuning'],freqs)
        coordinates = zip(active_cell_dict[cell]['xs'],active_cell_dict[cell]['ys'])

        for coord in coordinates:
            x,y = coord

            for i in range(3):
                bw_anat[x,y,i] = color[i]
    
    # bw_anat = bw_anat[:,:,:-1]
    plt.imshow(bw_anat)
    plt.show()


if __name__=="__main__":
    main()