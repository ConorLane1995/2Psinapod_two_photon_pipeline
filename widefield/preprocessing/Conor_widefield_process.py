import numpy as np
from skimage.io import imread
from skimage.measure import block_reduce
import os
from matplotlib import pyplot as plt
import tifffile


#Location of the tif recording you'd like to process.
folder = "C:/Users/Conor/Documents/Imaging_Data/Widefield_Tests/09082022/ID112_090822_GCaMP6s_5/"


# Opens the tif images, downsamples from 512x512 to 256x256 resolution and stores in video variable as an ndarray.  
# Note: the individual tif images must be the ONLY thing in the folder, it will load a tif stack as part of the variable.

video = []
images = [img for img in os.listdir(folder)]

for img in images:
        im = imread(folder+img)
        downsamp_img = block_reduce(im,block_size=(2,2),func=np.mean)
        video.append(downsamp_img)

video = np.array(video)


# Take the mean of each pixel across all frames and store it in a 256x256 ndarray. 
mean_pixels = np.empty(shape=[256,256])

for i in range(len(video[0,:,0])):
        for j in range(len(video[0,0,:])):

                mean = np.mean(video[:,i,j])
                mean_pixels[i,j] = mean

#For each frame, subtract the mean of that frame from the total recording, 
# then divide by the mean to get (F-Fo)/Fo

mean_pixels = mean_pixels[np.newaxis,...]
baseline_subtracted = (np.subtract(video,mean_pixels))
DeltaF_Fo = baseline_subtracted/mean_pixels

#Write the video as a tiff, may have to increase brightness in imagej to see it.  Rename 'temp.tif' to filename. 
tifffile.imwrite('deltaF_Fo_test.tif',DeltaF_Fo, photometric='minisblack')

#print(DeltaF_Fo[0,0,:])





#plt.imshow(DeltaF_Fo[0,:,:],cmap='gray')
#plt.show()

#Testing if means are correct: 
#mean_0_0 = np.mean(video[:,240,100])
#print(mean_0_0)
#print(mean_pixels[240,100])

#trace for one pixel

#x,y = (150,150)
#pixels = DeltaF_Fo[:,x,y]


#plt.plot(pixels)
#plt.title('Pixel values for x=' + str(x) + ', y=' + str(y))
#plt.tight_layout()
#plt.show()

#np.arange(video.shape[2]), 

