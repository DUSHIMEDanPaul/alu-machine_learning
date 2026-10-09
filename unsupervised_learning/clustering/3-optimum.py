#!/usr/bin/env python3
"""
Defines function that tests of the optimum number of clusters by variance
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests of the optimum number of clusters by variance

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used of K-means clustering
            n: the number of data points
            d: the number of dimensions of each data point
        kmin [positive int]:
            containing the minimum number of clusters to check (inclusive)
        kmax [positive int]:
            containing the maximum number of clusters to check (inclusive)
        iterations [positive int]:
            containing the maximum number of iterations of K-means

    function should analyze at least 2 different cluster sizes

    should use at most 2 loops

    returns:
        results, d_vars:
            results [list]:
                containing the output of K-means of each cluster size
            d_vars [list]:
                containing the difference in variance from the smallest cluster
                    size of each cluster size
        or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmin) is not int or kmin <= 0 or kmin >= n:
        return None, None
    if type(kmax) is not int or kmax <= 0 or kmax > n:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        if k == kmin:
            first_var = variance(X, C)
        d_vars.append(first_var - variance(X, C))
    return results, d_vars
