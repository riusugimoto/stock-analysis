import numpy as np
import matplotlib.pyplot as plt
import utils


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        W = np.reshape(
            weights_flat[counter : counter + W_size],
            (layer_sizes[i + 1], layer_sizes[i]),
        )
        counter += W_size
        weights.append(W)

    return weights


class NeuralNetRegressor:
    def __init__(
        self,
        hidden_layer_sizes,
        X=None,
        y=None,
        max_iter=20_000,
        learning_rate=0.001,
        plot_iter=True,
        plot_end=True,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.plot_iter = plot_iter
        self.plot_end = plot_end
        if X is not None and y is not None:
            self.fit(X, y)

    def loss_and_grad(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)
        activations = [X]
        for W in weights:
            Z = X @ W.T
            X = np.tanh(Z)
            activations.append(X)

        yhat = Z
        f = 0.5 * np.sum((yhat - y) ** 2)
        grad = yhat - y
        grad_W = grad.T @ activations[-2]
        g = [grad_W]
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W = weights[i]
            grad = grad @ W
            grad = grad * (1 - activations[i] ** 2)
            grad_W = grad.T @ activations[i - 1]
            g = [grad_W] + g

        g = flatten_weights(g)
        return (f, g)

    def fit(self, X, y):
         # Standardize features and targets
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean) / self.X_std
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y = (y - self.y_mean) / self.y_std
        if y.ndim == 1:
            y = y[:, None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        scale = 1
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            weights.append(W)

        if self.plot_iter:
            _, ax = plt.subplots()

        weights_flat = flatten_weights(weights)
        for i in range(self.max_iter):
            subset = np.random.choice((X.shape[0]), size=1, replace=False)
            loss, step_gradient = self.loss_and_grad(weights_flat, X[subset], y[subset])
            weights_flat = weights_flat - self.learning_rate * step_gradient
            self.weights = unflatten_weights(weights_flat, self.layer_sizes)

            if self.plot_iter and i % 500 == 0:
                print(f"Iteration {i:>10,}: loss = {loss:>6.3f}")

                ax.clear()
                self.plot_regression(X, y, ax=ax)
                plt.pause(0.01)

        if self.plot_end:
            fn = "regression-net.pdf" if self.plot_end is True else self.plot_end
            self.plot_regression(X, y, filename=fn)













    def plot_regression(self, X, y, filename=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        mesh = np.linspace(-10, 10, 1000)[:, None]

        d = X.shape[1]
        if d == 1:
            X_col = 0
        elif d == 2:
            # is one dimension a bias variable?
            ranges = np.ptp(X, axis=0)
            if np.any(ranges == 0):
                const_j = (ranges == 0).nonzero()[0][0]
                val = X[0, const_j]
                const = np.repeat(val, mesh.shape[0])

                if const_j == 0:
                    mesh = np.c_[const, mesh]
                    X_col = 1
                else:
                    mesh = np.c_[mesh, const]
                    X_col = 0
            else:
                raise ValueError("I'm confused, d=2 and neither is constant")
        else:
            raise ValueError("I'm confused, dimension too high")

        ax.plot(mesh[:, X_col], self.predict(mesh), c="g")
        ax.scatter(X[:, X_col], y, s=10)

        if filename is not None:
            ax.get_figure().savefig("../figs/" + filename)

    def predict(self, X):
        for W in self.weights:
            Z = X @ W.T
            X = np.tanh(Z)
        return Z
