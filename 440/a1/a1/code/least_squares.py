import numpy as np
from utils import euclidean_dist_squared


class LeastSquares:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        return X @ self.w


class LeastSquaresBias:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
         # add a bias meaning augmenting X with a column of ones 
        X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.linalg.solve(X_augmented.T @ X_augmented, X_augmented.T @ y)


    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_augmented @ self.w


def gaussianRBF_feats(X, bases, sigma):
    # Not mandatory, but might be nicer to implement this separately.
    raise NotImplementedError()


class LeastSquaresRBFL2:
    def __init__(self, X=None, y=None, lam=1, sigma=1):
        self.lam = lam
        self.sigma = sigma
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
         self.train_X = X
         new_x = euclidean_dist_squared(X, X)
         Z = np.exp(-new_x/(2*self.sigma**2))
         Z_augmented = np.hstack([Z, np.ones((Z.shape[0], 1))])
         I = np.identity(Z_augmented.shape[1])

         self.w = np.linalg.solve(Z_augmented.T @ Z_augmented + self.lam*I, Z_augmented.T @ y)
        

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!") 
        new_x = euclidean_dist_squared(X, self.train_X)
        Z = np.exp(-new_x/(2*self.sigma**2))
        Z_augmented = np.hstack([Z, np.ones((Z.shape[0], 1))])
       
        return Z_augmented @ self.w
        
