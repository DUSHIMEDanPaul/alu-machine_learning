#!/usr/bin/env python3
"""
Defines function that performs agglomerative clustering on a dataset
"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        dist [positive int]:
            the maximum cophenetic distance for all clusters

    performs agglomerative clustering with Ward linkage

    displays the dendrogram with each cluster displayed in a different color

    returns:
        clss [numpy.ndarray of shape (n,)]:
            containing the cluster indices for each data point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
