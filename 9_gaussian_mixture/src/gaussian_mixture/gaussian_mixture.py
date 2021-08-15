import numpy as np

class GaussianMixture(object):
    """
    construct mixture of Gaussians

    Parameters
    ----------
    n_components : int
    mu : (n_components, ndim) np.ndarray
    cov : (ncomponents, ndim, ndim) np.ndarray
    coef : (n_components,) np.ndarray
    """
    def __init__(self, 
                n_component,
                mu=None,
                cov=None,
                coef=None):
        self.n_component = n_component
        self.mu = mu
        self.cov = cov
        self.coef = coef
    
    def fit(self, X, iter_max=10):
        ndim= np.size(X, 1)
        self.coef= np.ones(self.n_component) / self.n_component
        self.mu = np.random.uniform(X.min(), X.max(), (ndim, self.n_component))
        self.cov = np.repeat(10 * np.eye(ndim), self.n_component).reshape(ndim, ndim, self.n_component)

        for i in range(iter_max):
            params = np.hstack((self.coef.ravel(), self.mu.ravel(), self.cov.ravel()))
            resps = self._expectation(X)
            self._maximization(X, resps)
            if np.allclose(params, np.hstack((self.coef.ravel(), self.mu.ravel(), self.cov.ravel()))):
                break

        else:
            print("parameters may not have converged")

    def _expectation(self, X):
        resps = self.coef * self.gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps     

    def _maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.coef = Nk / len(X)
        self.mu = X.T.dot(resps) / Nk
        diffs = X[:, :, None] - self.mu
        self.covs = np.einsum('nik,njk->ijk', diffs, diffs * np.expand_dims(resps, 1))/ Nk
 
    def gauss(self, X):
        ndim=np.size(X,1)
        precisions = np.linalg.inv(self.cov.T).T
        diffs = X[:, :, None] - self.mu
        assert diffs.shape == (len(X), ndim, self.n_component)
        exponents = np.sum(np.einsum('nik,ijk->njk', diffs, precisions) * diffs, axis=1)
        assert exponents.shape == (len(X), self.n_component)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.cov.T).T * (2 * np.pi) ** ndim)
    
    def predict_proba(self, X):
        gauss = self.coef * self.gauss(X)
        return np.sum(gauss, axis=1)
    
    def classify(self, X):
        joint_prob = self.coef * self.gauss(X)
        return np.argmax(joint_prob, axis=1)