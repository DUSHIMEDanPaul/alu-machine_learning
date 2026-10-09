#!/usr/bin/env python3
"""
Defines function that calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters

    returns:
        pi, m, S, clss, bic:
            pi [numpy.ndarray of shape (k,)]:
                containing the cluster priors
            m [numpy.ndarray of shape (k, d)]:
                containing the centroid means
            S [numpy.ndarray of shape (k, d, d)]:
                containing the covariance matrices
            clss [numpy.ndarray of shape (n,)]:
                containting the cluster indices for each data point
            bic [numpy.ndarray of shape (kmax - kmin + 1)]:
                containting the BIC value for each cluster size tested
    """
    g = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    return pi, m, S, clss, bic
