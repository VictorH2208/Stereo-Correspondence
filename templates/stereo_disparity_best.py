import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd, gradient= False):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """

    # From the image in part 1, it is obvious that the output is noisy and requires further filtering.
    # The window size has been tuned to increase both speed and accuracy. To further enhance feature quality, 
    # a sharpening filter is applied in the beginning to the two images. Similarly, at the end to reduce 
    # the noise of the disparity map median and percentile filtering is used. The parameters are being tuned 
    # against the cones and teddy image. In addition, to increase the accuracy of the disparity map, the 
    # gradients are being used. It occurs that with a weighted sum of the sad scores, there exist a good pbad 
    # value and a low RMSE. However, Gradient cannot be used due to time constraints.

    Id = np.zeros(Il.shape)
    # Sharpening filter
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    Il_sharp = convolve(Il, sharp_kernel, mode='constant')
    Ir_sharp = convolve(Ir, sharp_kernel, mode='constant')

    window_size = 7
    half_window = window_size // 2

    if gradient:
        # Compute gradients outside the loop
        Il_grad = compute_gradients(Il_sharp)
        Ir_grad = compute_gradients(Ir_sharp)

        # Pad images and gradients
        pad_Il = np.pad(Il_sharp, half_window, mode='edge')
        pad_Ir = np.pad(Ir_sharp, half_window, mode='edge')
        pad_Il_grad = np.pad(Il_grad, half_window, mode='edge')
        pad_Ir_grad = np.pad(Ir_grad, half_window, mode='edge')

        for y in range(bbox[1, 0], bbox[1, 1] + 1):
            for x in range(bbox[0, 0], bbox[0, 1] + 1):
                best_d = 0
                best_score = np.inf

                for d in range(-maxd, maxd):
                    if 0 <= x + d < Il.shape[1]:
                        score = compute_sad(pad_Il, pad_Ir, pad_Il_grad, pad_Ir_grad, x, y, d, half_window)
                        if score < best_score:
                            best_score = score
                            best_d = abs(d)

                Id[y, x] = best_d
    else:
        # Pad the left and right images
        padIl = np.pad(Il_sharp, half_window, mode = 'edge')
        padIr = np.pad(Ir_sharp, half_window, mode = 'edge')
        new_length, new_width = np.shape(padIl)

        # Loop through every pixel in the bounding box
        for i in range(bbox[1, 0], bbox[1, 1] + 1):
            for j in range(bbox[0, 0], bbox[0, 1] + 1):
                best_score = np.inf
                best_d = 0
                # Extract the window in the left image
                left = padIl[i : i + window_size, j : j + window_size]

                # Iterate over possible disparities within the specified range
                for d in range(-maxd, maxd):
                    # Ensure that the disparity does not go out of the image bounds
                    if (half_window - 1) < (j + d + half_window) < (new_width - half_window):
                        # Extract the corresponding window in the right image
                        right = padIr[i : i + window_size, j + d : j + d + window_size]

                        # Compute the Sum of Absolute Differences (SAD)
                        sad = np.sum(np.abs(left - right))
                        if sad < best_score:
                            best_score = sad
                            best_d = abs(d)

                # Assign the best disparity value to the disparity map
                Id[i, j] = best_d

    Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1] = median_filter(Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1], size=13)
    Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1] = percentile_filter(Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1], 40, size=17)

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id

def compute_gradients(image):
    grad = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2)
    return grad

def compute_sad(pad_Il, pad_Ir, pad_Il_grad, pad_Ir_grad, x, y, d, hw):
    # Extract windows and compute SAD
    window_Il = pad_Il[y:y + 2 * hw, x:x + 2 * hw]
    window_Ir = pad_Ir[y:y + 2 * hw, x + d:x + d + 2 * hw]
    window_Il_grad = pad_Il_grad[y:y + 2 * hw, x:x + 2 * hw]
    window_Ir_grad = pad_Ir_grad[y:y + 2 * hw, x + d:x + d + 2 * hw]
    
    ssd_intensity = np.sum(np.abs(window_Il - window_Ir))
    ssd_grad = np.sum(np.abs(window_Il_grad - window_Ir_grad))

    return ssd_intensity + 0.1 * ssd_grad