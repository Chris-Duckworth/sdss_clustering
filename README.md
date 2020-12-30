# sdss_clustering

Clustering of Sloan Digitial Sky Survey (SDSS) galaxy images (RGB) using transfer (from pre-trained convolutional neural networks) learning, and/or principal component analysis, and, hierarchical (agglomerative) clustering.

## Data

### Pre-Processing


## Principal Component Analysis 

As a baseline test of image similarity clustering, we first consider all pixel values (in each of the 3 colour channels) as distinct features (i.e. dimensions in the parameter space).
Since we are working with images of size (80, 80, 3), this corresponds to 19200 dimensions which is difficult to cluster directly. 
To compress information we apply a principal component analysis (pca) which transforms the parameter space, by considering a line orthogonal to the existing parameter dimensions, while minimising the distance of points from this line.
This equates to find dimensions that have maximal variance along them, hence compressing information while retaining a significant fraction of variance in the data.
Here, this can be thought of finding pixels that are highly correlated and linking them.

## CNN

Here we use a pre-trained convolutional neural network (CNN) in order to extract distinct features for the set of SDSS images. The structure and training of the CNN is described in this [repo](https://github.com/Chris-Duckworth/sdss_CNN).
In order to repurpose this network for the aim of clustering, we remove the final 4 `keras` defined layers (i.e. layers of fully connected nodes with dropout).
We now take the output from the flattened fully connect layer with 128 nodes, as a representation of our images. 

We find that a significant number of these nodes are practically redundant (i.e.) highly correlated, and hence, we can again compress information using pca. Here, we select only 20 compressed features from the CNN to represent the galaxy images. 
This PCA encapsulates 99.86% of the variance in the output features from the CNN.

## Clustering

### K-means

### Affinity Propagation

### Agglomerative