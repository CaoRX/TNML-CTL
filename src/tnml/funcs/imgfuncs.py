import numpy as np
from skimage.measure import block_reduce

def compress(img, px, py = None):
    if py is None:
        py = px
    return block_reduce(img, (px, py), np.average)
    