'''
sdss_prep - functions which pre-process sdss images for input into a CNN.
'''

import numpy as np
from PIL import Image

def process_image(path_to_file, shape=(80, 80)):
    '''
    Normalises raw RGB SDSS images to defined size
    and scales to 0-1 pixel value range for input 
    into the NN.

    ----------
    Parameters

    path_to_file : str
        Path (and filename) to load in image 
        using PIL. Assumes RGB (0-255 pixel values)
        
    shape : (height, width)
        Shape of output image.

    -------
    Returns

    image : np.array(shape)
        Processed image.
    '''
    im = Image.open(path_to_file)
    reshaped_im = np.array(im.resize(shape))
    
    return reshaped_im / 255.