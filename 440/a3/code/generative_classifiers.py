import numpy as np
from scipy import stats


class GDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        n, d = X.shape
        k = np.max(y) + 1  # assume ys are in {0, 1, ..., k-1}

        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError()


class TDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # import here, so rest of file doesn't need pytorch
        from student_t import MultivariateT

        raise NotImplementedError()

    def predict(self, Xtest):
        raise NotImplementedError()

        # NOTE: MultivariateT.log_prob() returns a torch tensor
        #       you might have to convert this to numpy, e.g.    t.detach().cpu().numpy()
