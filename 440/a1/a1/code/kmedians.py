import numpy as np
import matplotlib.pyplot as plt
from utils import plot2Dclusters

from kmeans import KMeans


# If you want, you could write this function to compute pairwise L1 distances
def l1_distances(X1, X2):
    D1 = np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
    return D1


class KMedians(KMeans):
    # We can reuse most of the code structure from KMeans, rather than copy-pasting,
    # by just overriding these few methods. Object-orientation!

      
    def get_assignments(self, X):
        D1 = l1_distances(X, self.w)
        return np.argmin(D1, axis=1)  
    

    def update_means(self, X, y):
        for k_i in range(self.k):
            matching = y == k_i
            if matching.any():
                   self.w[k_i] = np.median(X[matching], axis=0)
        


    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)

        loss = 0
        for i in range(X.shape[0]): 
            cluster_median = self.w[y[i]] 
            dist = np.sum(np.abs(X[i] - self.w[y[i]]))
            loss += dist
        
        return loss