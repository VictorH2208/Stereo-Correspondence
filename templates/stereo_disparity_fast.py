import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):

    # Define the size of the window
    window_size = 20

    # Initialize the disparity map with zeros
    Id = np.zeros(np.shape(Il)) 

    # Pad the left and right images
    padIl = np.pad(Il, 10, mode = 'edge')
    padIr = np.pad(Ir, 10, mode = 'edge')
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
                if (j + d + 10) < (new_width - 10) and (j + d + 10) > 9:
                    # Extract the corresponding window in the right image
                    right = padIr[i : i + window_size, j + d : j + d + window_size]

                    # Compute the Sum of Absolute Differences (SAD)
                    sad = np.sum(np.abs(left - right))
                    if sad < best_score:
                        best_score = sad
                        best_d = abs(d)

            # Assign the best disparity value to the disparity map
            Id[i, j] = best_d
                    
    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id


