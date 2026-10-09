#!/usr/bin/env python3
"""
Defines function that determines the steady state probabilities of
a regular Markov Chain
"""


import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov Chain

    parameters:
        P [square 2D numpy.ndarray of shape (n, n)]:
            representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n: the number of state in the Markov Chain

    returns:
        [a numpy.ndarray of shape (1, n)]:
            representing the steady state probabilities
        or None on failure
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return None
    # each row of P must be a probability distribution
    if not np.isclose(np.sum(P, axis=1), 1).all():
        return None
    # P is regular if some power of P has only positive entries;
    # checking up to (n - 1) ** 2 + 1 is sufficient
    power = P
    for i in range((n - 1) ** 2 + 1):
        if (power > 0).all():
            break
        power = np.matmul(power, P)
    else:
        return None
    # steady state s satisfies s(P - I) = 0 with entries summing to 1
    A = np.vstack(((P - np.identity(n)).T, np.ones((1, n))))
    b = np.zeros((n + 1,))
    b[n] = 1
    steady = np.linalg.lstsq(A, b, rcond=None)[0]
    return steady[np.newaxis, :]
