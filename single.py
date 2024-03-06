# Import useful libraries
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
from IPython.display import display, Image
import queue
import time
from joblib import Parallel, delayed
import copy

# Display original image
image = iio.imread("junk.jpg")
fox = iio.imread("blurryFox.jpg")


# Convert image to array, print out the shape of array, and print out the entire array
img_matrix = iio.core.asarray(image)
fox_matrix = iio.core.asarray(fox)

foxR = copy.deepcopy(fox_matrix[:,:,0])
foxG = copy.deepcopy(fox_matrix[:,:,1])
foxB = copy.deepcopy(fox_matrix[:,:,2])

def applyFilterSingleThread(img: np.array, filter: np.array, stride) -> np.array:
    print(img.shape)
    tgt_sizeX = img.shape[0]
    tgt_sizeY = img.shape[1]

    # To simplify things
    k = filter.shape[0]

    padding_size = 1

    # Pad the input image with zeros
    padded_img = np.pad(img, ((padding_size, 0), (padding_size, 0)), mode='constant')

    # New size after padding
    padded_size = padded_img.shape[0]

    # 2D array of zeros
    convolved_img = np.zeros(shape=(padded_size, padded_size))

    # 2D array of zeros
    convolved_img = np.zeros(shape=(tgt_sizeX, tgt_sizeY))

    # Iterate here
    for i in range(0, tgt_sizeX - k + 1, stride):
        for j in range(0, tgt_sizeY - k + 1, stride):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+k, j:j+k]
            #print(mat)

            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, filter))
    return convolved_img

sharpeningFilter = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

#processed_imageR = applyFilterSingleThread(img = np.array(img_matrix[:,:,0]), filter = sharpeningFilter, stride = 1)
#processed_imageG = applyFilterSingleThread(img = np.array(img_matrix[:,:,1]), filter = sharpeningFilter, stride = 1)
#processed_imageB = applyFilterSingleThread(img = np.array(img_matrix[:,:,2]), filter = sharpeningFilter, stride = 1)

start_time = time.time()


foxR = applyFilterSingleThread(img = np.array(fox_matrix[:,:,0]), filter = sharpeningFilter, stride = 1)
foxG = applyFilterSingleThread(img = np.array(fox_matrix[:,:,1]), filter = sharpeningFilter, stride = 1)
foxB = applyFilterSingleThread(img = np.array(fox_matrix[:,:,2]), filter = sharpeningFilter, stride = 1)


#processed_imageR = (processed_imageR - processed_imageR.min()) / (processed_imageR.max() - processed_imageR.min())
#processed_imageG = (processed_imageG - processed_imageG.min()) / (processed_imageG.max() - processed_imageG.min())
#processed_imageB = (processed_imageB - processed_imageB.min()) / (processed_imageB.max() - processed_imageB.min())

#combined_image = np.dstack((processed_imageR, processed_imageG, processed_imageB))

combined_fox = np.dstack((foxR, foxG, foxB))
combined_fox = np.clip(combined_fox, 0, 255).astype(np.uint8)



#print(combined_image.shape)

#figR = plt.imshow(image)
#figR = plt.imshow(combined_image)
fig = plt.imshow(combined_fox)
end_time = time.time()
elapsed_time = end_time - start_time
print("ELAPSED_TIME:",elapsed_time, " seconds")
print("Output image size is ", combined_fox.shape)

iio.imwrite("result.jpg", combined_fox)