#!/usr/bin/env python3
"""
Defines function that finds the best number of clusters of a GMM using
the BIC
"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters of a GMM using BIC

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions of each data point
        kmin [positive int]:
            the minimum number of clusters to check (inclusive)
        kmax [positive int]:
            the maximum number of clusters to check (inclusive)
            if None, kmax should be set to maximum number of clusters possible
        iterations [positive int]:
            the maximum number of iterations of the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used to stop early
        verbose [boolean]:
            determines if you should print details about the algorithm

    should only use one loop

    returns:
        best_k, best_result, l, b
            best_k [positive int]:
                the best value of k based on its BIC
            best_result [tuple containing pi, m, S]:
                pi [numpy.ndarray of shape (k,)]:
                    contains cluster priors of the best number of clusters
                m [numpy.ndarray of shape (k, d)]:
                    contains centroid means of the best number of clusters
                S [numpy.ndarray of shape (k, d, d)]:
                    contains covariance matrices of best number of clusters
            l [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the log likelihood of each cluster size tested
            b [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the BIC value of each cluster size tested
                BIC = p * ln(n) - 2 * 1
                    p: number of parameters required of the model
                    n: number of data points used to create the model
                    l: the log likelihood of the model
        or None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmin) is not int or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    likelihoods = []
    bics = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        likelihoods.append(log_likelihood)
        # free parameters: priors, means, and symmetric covariances
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        bics.append(p * np.log(n) - 2 * log_likelihood)
    likelihoods = np.array(likelihoods)
    bics = np.array(bics)
    best = np.argmin(bics)
    return kmin + best, results[best], likelihoods, bics
