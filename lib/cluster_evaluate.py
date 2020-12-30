'''
cluster_evaluate 
'''

import matplotlib.pyplot as plt
import numpy as np


def plot_cluster(images, row_len=10):
    '''
    Given an np.array of images (dim : (n_img, x, y, channels) )
    this creates a subplot of all images. 
    '''
    
    n_img = images.shape[0]
    n_row = np.ceil(n_img / row_len).astype(int)
    
    fig, ax = plt.subplots(n_row, row_len, figsize=(20, 2 * n_row))
    
    for ind, im in enumerate(images):
        axis = ax.ravel()[ind]
        axis.imshow(im)
        
    # removing all axis labels
    for axis in ax.ravel():
        axis.axis('off')
    
    fig.subplots_adjust(wspace=0, hspace=0.05)
    return fig, ax
    
