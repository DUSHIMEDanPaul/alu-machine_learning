#!/usr/bin/env python3
"""
Defines function that calculates the probability density function of
a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset whose PDF should be calculated
            n: the number of data points
            d: the number of dimensions of each data point
        m [numpy.ndarray of shape (d,)]:
            contains the mean of the distribution
        S [numpy.ndarray of shape (d, d)]:
            contains the covariance of the distribution

    not allowed to use any loops
    not allowed to use the function numpy.diag or method numpy.ndarray.diagonal

    returns:
        P [numpy.ndarray of shape (n,)]:
            containing the PDF value of each data point
            all values in P should have a minimum value of 1e-300
        or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    n, d = X.shape
    if type(m) is not np.ndarray or m.shape != (d,):
        return None
    if type(S) is not np.ndarray or S.shape != (d, d):
        return None
    det = np.linalg.det(S)
    if det <= 0:
        return None
    inv = np.linalg.inv(S)
    diff = X - m
    # Mahalanobis term of each data point, computed without np.diag
    exponent = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
    coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    P = coefficient * np.exp(exponent)
    P = np.maximum(P, 1e-300)
    return P
