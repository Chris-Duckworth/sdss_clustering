'''
cluster_evaluate 
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

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


def save_cluster_plot(images, cluster_number, cluster_total, plot_path='./'):
    '''
    Given a set of images in a cluster, a cluster number (label), and the total number 
    of clusters (req. for filename format.) and the directory to be save to, this creates
    a complete grid of images for the cluster.
    '''

    fig, ax = plot_cluster(images)
    # adding title to first subplot - suptitle for a variable number of rows moves relatively to images.
    ax.ravel()[0].set_title('Cluster {}, ({} galaxies)'.format(cluster_number, images.shape[0]), x=5, fontsize=35, ha='center')
    plt.savefig(plot_path+'cluster_' + str(cluster_number) + '_of_' + str(cluster_total) +'.pdf',
                format='pdf') ;
    return 


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    leaves_list = hierarchy.leaves_list(linkage_matrix)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)