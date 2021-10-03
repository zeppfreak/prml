import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gaussian_mixture.gaussian_mixture import GaussianMixture

X = np.random.normal(size=(10, 2))
X += np.array([-5, -5])


@pytest.fixture
def gm():
    n_component = 3
    ndim = np.size(X, axis=1)
    mu = np.random.uniform(X.min(), X.max(), (ndim, n_component))
    cov = np.repeat(10 * np.eye(ndim), n_component).reshape(ndim, ndim, n_component)
    coef = np.random.uniform(0.0, 1.0, n_component)
    coef /= coef.sum()

    gm = GaussianMixture(n_component, mu=mu, cov=cov, coef=coef)
    return gm


def test_gauss(gm):
    print(gm.gauss(X))
    assert gm.gauss(X).shape == (len(X), gm.n_component)


def test_expectation(gm):
    resps = gm._expectation(X)
    print(resps)
    assert resps.shape == (len(X), gm.n_component)


def test_maximization(gm):
    print("mu before: {}".format(gm.mu))
    print("cov before: {}".format(gm.cov))
    print("coef before: {}".format(gm.coef))
    resps = gm._expectation(X)
    gm._maximization(X, resps)
    print("mu after: {}".format(gm.mu))
    print("cov after: {}".format(gm.cov))
    print("coef after: {}".format(gm.coef))
