import numpy as np
from .preprocessing import invert_img

__all__ = [
    'compute_path',
    'draw_paths',
    'erase_path'
]

# Computes staff's space and line height from a binary image
def _compute_heights(img):
    width = img.shape[1]
    # Adds a background (0's) line to the top and bottom of the image
    #   so all columns start and end with background,
    #   then all columns are flattened to a single vector to improve performance
    img = np.vstack((np.zeros(width), img, np.zeros(width))).flatten('F')
    # Computes the indices where value changes from background to ink or viceversa
    # The first one (background) is lost
    changes = np.where(img[:-1] != img[1:])[0]
    # Computes the thickness of each object (background or ink)
    #   by substracting the indices where changes happen
    # The last one (background) is lost
    heights = changes[1:] - changes[:-1]
    # Spaces are at odd positions (1, 3, ...)
    #   and lines at even positions (0, 2, ...)
    #   because the first occurrence of background has been discarded
    space_height = np.argmax(np.bincount(heights[1::2]))
    line_height = np.argmax(np.bincount(heights[::2]))
    return space_height, line_height

# TODO: modify weights based on vertical run length and paralel staffs
def _compute_weights(img):
    height, width = img.shape
    # Extended image to make it easier to compute weights
    #   for the first column and first and last rows,
    #   adds a left column of zeros and a top and bottom rows of weight_max
    # weight_max is used to prevent paths coming from outside the image
    img_ext = np.vstack((np.ones((1, width + 1)) * 3, #weight_max,
                         np.hstack((np.zeros((height, 1)),
                                    img)),
                         np.ones((1, width + 1)) * 3)) #weight_max))
    img_ext = np.uint8(img_ext)
    # Weight matrix
    W = np.zeros((height, width, 3), dtype=int)
    # Diagonal edges weight 6 if there is a black pixel and 12 if not
    W[:, :, 0] = (np.logical_and(img, img_ext[:-2, :-1]) + 1) * 6
    W[:, :, 2] = (np.logical_and(img, img_ext[2:, :-1]) + 1) * 6
    # Straight edges weight 4 if there is a black pixel and 8 if not
    W[:, :, 1] = (np.logical_and(img, img_ext[1:-1, :-1]) + 1) * 4
    # Edges from outside the image (top or bottom) weight more
    #   than the most expensive possible path inside the image
    W[0, :, 0] = width * 12 + 1
    W[-1, :, 2] = width * 12 + 1
    return W

def _compute_costs(img):
    height, width = img.shape
    W = _compute_weights(img)                # Weight matrix
    C = np.zeros((height+2, width+1))        # Cost matrix
    P = np.zeros((height, width), dtype=int) # Path matrix
    # Compute costs and paths
    for column in range(1,width+1):
        C_step = np.array([C[:-2, column-1] + W[:, column-1, 0],
                           C[1:-1, column-1] + W[:, column-1, 1],
                           C[2:, column-1] + W[:, column-1, 2]])
        # Minimum cost from left-top, left, or left-bottom
        C[1:-1, column] = np.amin(C_step, axis=0)
        # Direction of the minimum cost
        P[:, column-1] = np.argmin(C_step, axis=0) - 1
    C = C[1:-1,1:]
    nodes = np.arange(height, dtype=int)
    for column in range(width-1, 0, -1):
        directions = P[:, column].copy()
        P[:, column] = nodes
        nodes = nodes + directions
        P[:, column-1] = P[nodes, column-1]
    P[:, 0] = nodes
    return C, P

def erase_path(img, path, line_height):
    height, width = img.shape
    for x in range(0, width):
        for i in range(-int(line_height/2), int(line_height/2 + 1)):
            y = path[x] + i
            if (y >= 0) and (y < height):
                img[y, x] = 255
    return img

def _trim_paths(paths, img):
    x = np.arange(paths[0].size)
    trimmed_paths = np.array([])
    for idx, path in enumerate(paths):
        line = img[path, x]
        print(line)
        first = np.argmax(line == 0)
        last = np.argmax(line[::-1] == 0)
        print(x[first: last-1])
        #trimmed_paths.append(np.array((x[first: last+1], path[first: last+1])).T)
        np.append(trimmed_paths, np.array((x[first: last+1], path[first: last+1])).T)
    return trimmed_paths

def compute_path(img, invert=False):
    if invert:
        img = invert_img(img)
    # Convert image to array of ints (0's and 1's) to improve performance
    img = np.uint8(img / np.max(img))
    height, width = img.shape
    # Computes costs from left to right and from right to left
    C_lr, P_lr = _compute_costs(img)
    C_rl, P_rl = _compute_costs(np.fliplr(img))
    # Only the paths that start and end at the same points in both
    #   directions are stable paths
    start = P_lr[:, 0]
    end = P_rl[P_lr[:, width-1], width-1]
    stable_paths = np.argwhere(np.equal(start, end)).flatten()
    return _trim_paths(P_lr[stable_paths], img)

def draw_paths(img, paths, line_height=1):
    img = np.copy(img)
    height, width = img.shape[:2]
    x = np.array([np.arange(width, dtype=int)] * paths.shape[0]).flatten()
    for i in range(-int(line_height/2), int(line_height/2+1)):
        img[paths.flatten().astype(int) + i, x] = [255, 0, 0]
    return img
