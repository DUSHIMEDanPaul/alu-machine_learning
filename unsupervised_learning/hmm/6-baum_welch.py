#!/usr/bin/env python3
"""
Defines function that performs the Baum-Welch algorithm for Hidden Markov Model
"""


import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
            T: number of observations
        Transition [2D numpy.ndarray of shape (M, M)]:
            contains the initialized transition probabilities
            M: the number of hidden states
        Emission [numpy.ndarray of shape (M, N)]:
            contains the initialized emission probabilities
            N: number of output states
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the initialized starting probabilities
        iterations [positive int]:
            the number of times expectation-maximization should be performed

    returns:
        the converged Transition, Emission
        or None, None on failure
    """
    # check that Observations is the correct type and dimension
    if type(Observations) is not np.ndarray or \
            len(Observations.shape) != 1:
        return None, None
    # save T from Observations' shape
    T = Observations.shape[0]
    # check that Transition is the correct type and dimension
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    # save M and check that Transition is square
    M, M_check = Transition.shape
    if M != M_check:
        return None, None
    # check that Emission is the correct type and dimension
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    # check that Emission's dimension matches N from Transition and save N
    M_check, N = Emission.shape
    if M_check != M:
        return None, None
    # check that Initial is the correct type and dimension
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    # check that Initial's dimensions match (M, 1)
    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None
    # check that iterations is a positive int
    if type(iterations) is not int or iterations < 1:
        return None, None
    for i in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        # xi[i, j, t]: probability of state i at t and state j at t + 1
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            numerator = alpha[:, t, np.newaxis] * Transition * \
                Emission[:, Observations[t + 1]] * beta[:, t + 1]
            xi[:, :, t] = numerator / np.sum(numerator)
        # gamma[i, t]: probability of state i at t
        gamma = alpha * beta / np.sum(alpha * beta, axis=0)
        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :T - 1], axis=1)[:, np.newaxis]
        Emission = np.zeros((M, N))
        for k in range(N):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = Emission / np.sum(gamma, axis=1)[:, np.newaxis]
    return Transition, Emission
