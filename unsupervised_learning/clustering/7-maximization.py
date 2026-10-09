#!/usr/bin/env python3
"""
Defines function that calculates the maximization step in the EM algorithm
of a Gaussian Mixture Model
"""


import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm of a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions of each data point
        g [numpy.ndarray of shape (k, n)]:
            containing the posterior probabilities of each data point
                in the cluster

    should only use one loop

    returns:
        pi, m, S:
            pi [numpy.ndarray of shape (k,)]:
                containing the updated priors of each cluster
            m [numpy.ndarray of shape (k, d)]:
                containing the updated centroid means of each cluster
            S [numpy.ndarray of shape (k, d, d)]:
                containing the updated covariance matrices of each cluster
        or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    n, d = X.shape
    if type(g) is not np.ndarray or len(g.shape) != 2 or g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    # posteriors of each data point must sum to 1 across clusters
    if not np.isclose(np.sum(g, axis=0), np.ones((n,))).all():
        return None, None, None
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        g_sum = np.sum(g[i])
        pi[i] = g_sum / n
        m[i] = np.matmul(g[i], X) / g_sum
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / g_sum
    return pi, m, S
