# Import useful libraries
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
from IPython.display import display, Image
import queue
import time
from joblib import Parallel, delayed
import copy
import cupy as cp

# Display original image
image = iio.imread("junk.jpg")
fox = iio.imread("blurryFox.jpg")


# Convert image to array, print out the shape of array, and print out the entire array
img_matrix = iio.core.asarray(image)
fox_matrix = iio.core.asarray(fox)

foxR = copy.deepcopy(fox_matrix[:,:,0])
foxG = copy.deepcopy(fox_matrix[:,:,1])
foxB = copy.deepcopy(fox_matrix[:,:,2])

#print("Shape of the array is " + str(img_matrix.shape))

def applyMultiThread(img: cp.array, filter: cp.array, stride) -> cp.array:
    tgt_sizeX = img.shape[0]
    tgt_sizeY = img.shape[1]
    k = filter.shape[0]

    convolved_img = cp.zeros((tgt_sizeX, tgt_sizeY))

    for i in range(0, tgt_sizeX - k + 1, stride):
        for j in range(0, tgt_sizeY - k + 1, stride):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = cp.sum(cp.multiply(mat, filter))

    return convolved_img

def doMultiThread(R: np.array, G: np.array, B: np.array):
    sharpeningFilter1 = cp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    sharpeningFilter2 = cp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    sharpeningFilter3 = cp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    results = Parallel(n_jobs=3)(delayed(applyMultiThread)(cp.array(img), cp.array(filt), 1) for img, filt in zip([R, G, B], [sharpeningFilter1, sharpeningFilter2, sharpeningFilter3]))

    combined_image = cp.dstack((results[0], results[1], results[2]))
    combined_image = cp.asnumpy(combined_image) # Convert back to NumPy array for plotting
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    return combined_image


# Use cupy.array for GPU compatibility
foxR_gpu = cp.array(foxR)
foxG_gpu = cp.array(foxG)
foxB_gpu = cp.array(foxB)

start_time = time.time()
result = doMultiThread(foxR_gpu, foxG_gpu, foxB_gpu)
end_time = time.time()
elapsed_time = end_time - start_time
print("ELAPSED_TIME:", elapsed_time, " seconds")

iio.imwrite("result.jpg", result)