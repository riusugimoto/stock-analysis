import numpy as np
from scipy.special import logsumexp

from kmeans import KMeans


class NaiveNaiveBayes:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

        n, self.d = X.shape

        # categorical MLE for y:
        self.y_probs = np.bincount(y) / n

        # fit a Bernoulli for each X variable independently
        self.p_x = np.mean(X, axis=0)

    def predict(self, X):
        t, d = X.shape
        assert d == self.d

        # in our "naive naive bayes", y is totally independent of X
        # so no real reason to even look at the data we get!
        return np.repeat(np.argmax(self.y_probs), t)


class NaiveBayes:
    def __init__(self, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

        # num_classes is just a "backup" to pass k in case you don't actually see
        # any from the highest class

        n, self.d = X.shape

        # categorical MLE for y:
        self.y_probs = np.bincount(y) / n
        self.num_classes = k = len(self.y_probs)
        self.theta = np.zeros((k, self.d))
  
        # Calculate conditional probabilities for each class
        for c in range(k):
            X_c = X[y == c] #get each row of X matrix
            #MAP from lecture slide 2 p24
            self.theta[c] = (np.sum(X_c, axis=0) + self.prior_alpha-1) / (X_c.shape[0] + self.prior_alpha + self.prior_beta-2)
 

    def predict(self, X):
        n, self.d = X.shape
        k = len(self.y_probs)

        # Calculate log probabilities for each class
        log_probs = np.zeros((n, k))
        for c in range(k):
            log_probs[:, c] = (
                np.sum(np.log(self.theta[c]) * X, axis=1) + 
                np.sum(np.log(1 - self.theta[c]) * (1 - X), axis=1) +
                np.log(self.y_probs[c])
            )

 
        return np.argmax(log_probs, axis=1)
        


class VQNB:
    def __init__(self, k, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.k = k
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

       
        n, self.d = X.shape
        self.y_probs = np.bincount(y) / n
        self.num_classes = len(self.y_probs)
        self.theta = np.zeros((self.num_classes, self.k, self.d))
        self.cluster_probs = np.zeros((self.num_classes, self.k))

        for c in range(self.num_classes):
            X_c = X[y == c]
            kmeans = KMeans(k=self.k)
            kmeans.fit(X_c)
            clusters = kmeans.get_assignments(X_c)

            # Calculate cluster probabilities p(z|y)
            self.cluster_probs[c] = (np.bincount(clusters, minlength=self.k) + self.prior_alpha) / (len(clusters) + self.k * self.prior_alpha)

       
            for z in range(self.k):
                X_cz = X_c[clusters == z]  
                
                self.theta[c, z] = (np.sum(X_cz, axis=0) + self.prior_alpha) / (X_cz.shape[0] + self.prior_alpha + self.prior_beta)

                # if X_cz.size > 0:
                #     self.theta[c, z] = (np.sum(X_cz, axis=0) + self.prior_alpha) / (X_cz.shape[0] + self.prior_alpha + self.prior_beta)
                # else:
                #     self.theta[c, z] = self.prior_alpha / (self.prior_alpha + self.prior_beta)

                        

    def predict(self, X):
        n, d = X.shape
        num_classes = self.theta.shape[0]
        log_probs = np.zeros((n, num_classes))

        # for c in range(num_classes):
        #     log_prob_y = np.log(self.cluster_probs[c])
        #     log_prob_x_given_y_z = np.sum(np.log(self.theta[c]) * X[:, np.newaxis, :], axis=2) + \
        #                            np.sum(np.log(1 - self.theta[c]) * (1 - X[:, np.newaxis, :]), axis=2)
        #     # Marginalize over clusters using logsumexp for numerical stability
        #     log_probs[:, c] = np.log(self.y_probs[c]) + logsumexp(log_prob_y + log_prob_x_given_y_z, axis=1)

        for i in range(n):
            for c in range(num_classes):
                # Compute log P(y=c)
                log_prob_y = np.log(self.y_probs[c])

               
                log_prob_x_given_y_z = np.zeros(self.k)

                for z in range(self.k):
                    # Compute log P(z|y) for the cluster
                    log_prob_z_given_y = np.log(self.cluster_probs[c, z])

                    # Compute log P(x|y,z) for all features
                    log_prob_x_given_y_z[z] = (
                        np.sum(np.log(self.theta[c, z]) * X[i]) +
                        np.sum(np.log(1 - self.theta[c, z]) * (1 - X[i]))
                    )

                # now put them together  log P(y) + logsumexp(log P(z|y) + log P(x|y,z))

                log_probs[i, c] = log_prob_y + logsumexp(log_prob_z_given_y + log_prob_x_given_y_z)
    
        return np.argmax(log_probs, axis=1)
