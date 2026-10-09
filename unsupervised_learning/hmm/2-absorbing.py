#!/usr/bin/env python3
"""
Defines function that determines if the Markov Chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    Determines if the Markov Chain is absorbing

    parameters:
        P [square 2D numpy.ndarray of shape (n, n)]:
            representing the standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n: the number of state in the Markov Chain

    returns:
        True, if absorbing
        False, if not absorbing or on failure
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return False
    # absorbing states can never be left
    absorbing_states = np.isclose(np.diag(P), 1)
    if not absorbing_states.any():
        return False
    # spread reachability from absorbing states backwards through P
    can_absorb = absorbing_states
    for i in range(n):
        reaches = np.any(P[:, can_absorb] > 0, axis=1) | can_absorb
        if (reaches == can_absorb).all():
            break
        can_absorb = reaches
    return bool(can_absorb.all())
