import numpy as np
import itertools


def polynomial_basis(s, n):
    # return the (n+1)**k polynomial features of state s
    features = []
    k = len(s)
    for exponents in itertools.product(range(n+1), repeat=k):
        xi = 1
        for si, ci in zip(s, exponents):
            xi *= si**ci
        features.append(xi)

    return np.asarray(features)


def gradient_mc_poly(samples, n, alpha, n_passes):
    # return the learned weight vector
    if not samples:
        return []

    k = len(samples[0][0])
    w = np.zeros(((n+1)**k,))

    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)
            v = (w * x).sum()
            w += alpha * (G - v) * x

    return w
