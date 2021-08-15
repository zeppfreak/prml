import pytest
import numpy as np

from bayesian_curve_fitting import PolynomialFeatures, BayesianRegression, create_toy_data

@pytest.fixture
def bayesian_regression_fixture():
    br = BayesianRegression(alpha=0.1, beta=0.25)
    return br

def test_fit(bayesian_regression_fixture):
    degree = 5
    x = np.linspace(0.1, 1.0, 10)
    t = np.sin(2 * np.pi * x)
    print("x: {}".format(x))
    print("t: {}".format(t))

    features = [x ** i for i in np.arange(degree + 1)]
    X = np.array(features).transpose()
    print("X: {}:".format(X))

    bayesian_regression_fixture.fit(X, t)
    print("w_mean: {}".format(bayesian_regression_fixture.w_mean))
     