#!/usr/bin/env python3
"""
Defines function that perfoms the expectation maximization (EM)
of a Gaussian Mixture Model
"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Runs the expectation maximization (EM) of a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions of each data point
        k [positive int]:
            the number of clusters
        iterations [positive int]:
            the maximum number of iterations of the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used to stop early
            if the difference is less than or equal to tol, stop the algorithm
        verbose [boolean]:
            determines if you should print details about the algorithm
            if true: print 'Log Likelihood after {i} iterations: {l}'
                every 10 iterations and after the last iteration
            {i}: number of iterations of the EM algorithm
            {l}: log likelihood, rounded to 5 decimal places

    should only use one loop

    returns:
        pi, m, S, g, l:
            pi [numpy.ndarray of shape (k,)]:
                containing the priors of each cluster
            m [numpy.ndarray of shape (k, d)]:
                containing the centroid means of each cluster
            S [numpy.ndarray of shape (k, d, d)]:
                containing the covariance matrices of each cluster
            g [numpy.ndarray of shape (k, n)]:
                containing probabilities of each data point in each cluster
            l [float]:
                log likelihood of the model
        or None, None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    g, log_likelihood = expectation(X, pi, m, S)
    prev_likelihood = 0
    i = 0
    while i < iterations:
        if abs(log_likelihood - prev_likelihood) <= tol:
            break
        if verbose and i % 10 == 0:
            print('Log Likelihood after %d iterations: %s' %
                  (i, round(log_likelihood, 5)))
        prev_likelihood = log_likelihood
        pi, m, S = maximization(X, g)
        g, log_likelihood = expectation(X, pi, m, S)
        i += 1
    if verbose:
        print('Log Likelihood after %d iterations: %s' %
              (i, round(log_likelihood, 5)))
    return pi, m, S, g, log_likelihood
