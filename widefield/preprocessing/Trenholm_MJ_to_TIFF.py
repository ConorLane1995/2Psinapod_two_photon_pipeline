# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:58:23 2022

@author: Conor Lane
"""

import skvideo.io
import numpy as np
import tifffile

# Input filepath to .mj2 video
fn = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/wdf/RT_056/20220525/wdf/wdf_000_a11.mj2"

outputparameters = {'-pix_fmt' : 'gray16be'} # specify to import as uint16, otherwise it's uint8

vid = np.squeeze(skvideo.io.vread(fn,outputdict=outputparameters)) #  Read in video as numpy array 

tifffile.imwrite('temp.tif',vid, photometric='minisblack')   #  Write as TIFF, change 'temp.tif' to desired filename