#!/usr/bin/env python

# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

from functools import cache
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from kmeans import KMeans
from kmedians import KMedians
from utils import (
    test_and_plot,
    plot2Dclassifier,
    plot2Dclusters,
    load_dataset,
    main,
    handle,
    run,
)
from least_squares import LeastSquares, LeastSquaresBias, LeastSquaresRBFL2
from mmul_order import mmul_order


@handle("kmeans")
def q_kmeans():
    (X,) = load_dataset("cluster_data", "X")
    model = KMeans(X, 4, plot=True)


def best_fit(X, k, reps=50, cls=KMeans):
    # Fit a cls() reps number of times, and return the best one according to model.loss()
    # Use  cls(X, k, plot=False, log=False)  to fit a model,
    # so it'll work for both KMeans and KMedians.
    # (Passing plot=False makes it run a *lot* faster, and log=False avoids a ton of clutter.)

    best_loss = np.inf
    best_model = None
    for i in range(reps): 
        model = cls(X, k, plot=False, log=False)
        loss =  model.loss(X)
        if best_loss > loss:
            best_loss = loss
            best_model = model
        

    return best_model




@handle("kmeans-best")
def q_kmeans_best():
    (X,) = load_dataset("cluster_data", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(X, best_model.get_assignments(X), best_model.w, "kmeans-best.png")


@handle("kmeans-outliers")
def q_kmeans_outliers():
    (X,) = load_dataset("cluster_data_2", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(
        X, best_model.get_assignments(X), best_model.w, "kmeans-outliers.png"
    )


@handle("kmedians-outliers")
def q_kmedians_outliers():
    (X,) = load_dataset("cluster_data_2", "X")
    best_model = best_fit(X, k=4, cls=KMedians)
    plot2Dclusters(
        X, best_model.get_assignments(X), best_model.w, "kmedians-outliers.png"
    )


@handle("lsq")
def q_lsq():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquares(X, y)
    test_and_plot(model, X, y, filename="leastsquares.png")


@handle("lsq-bias")
def q_lsq_bias():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquaresBias(X, y)
    test_and_plot(model, X, y, filename="leastsquares-bias.png")


@handle("lsq-rbf")
def q_lsq_rbf():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquaresRBFL2(X, y)
    q_lsq_rbf_split()
    test_and_plot(model, X, y, filename="leastsquares-rbfl2.png")


def q_lsq_rbf_split():
    X, y = load_dataset("basis_data", "X", "y")
    lam_values = [0.01, 0.1, 1, 10, 100]
    sigma_values = [0.01, 0.1, 1, 10, 100]

    n = X.shape[0]
    fold_size = n // 2  

    best_lam = None
    best_sigma = None
    best_mse =  np.inf

    for lam in lam_values:
        for sigma in sigma_values:
            mse_values = []

            for i in range(2):
                mask = np.ones(n, dtype=bool)
                mask[i * fold_size: (i + 1) * fold_size] = False
              
                X_train =X[mask]
                y_train =y[mask]
                X_val = X[~mask]
                y_val = y[~mask]
                model = LeastSquaresRBFL2(X_train, y_train, lam, sigma)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)


                # Compute mse
                mse = np.mean((y_pred - y_val) ** 2)
                mse_values.append(mse)

            # Average mse for this combination of lamb and sigma
            avg_mse = np.mean(mse_values)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lam = lam
                best_sigma = sigma

    

    best_model = LeastSquaresRBFL2(X, y, best_lam, best_sigma)
    best_model.fit(X, y)
    test_and_plot(best_model, X, y, filename="bestleastsquares-rbfl2.png")




    





@handle("mmul-order")
def q_mmul_order():
    print("Test cases: ")

    print()
    dims = [90, 6, 12]
    cost, order = mmul_order(dims)
    print(f"dims: {dims}")
    print(f"Expected cost 6,480; order 1 x 2")
    print(f"Got      cost {cost:,}; order {order}")

    print()
    dims = [1, 1, 5, 1, 1]
    cost, order = mmul_order(dims)
    print(f"dims: {dims}")
    print(f"Expected cost 7; order 2 x 3; 2:3 x 4; 1 x 2:4")
    print(f"                    or 2 x 3; 1 x 2:3; 1:3 x 4")
    print(f"Got      cost {cost:,}; order {order}")

    print()
    dims = [12, 10, 5, 2, 12, 10000]
    cost, order = mmul_order(dims)
    print(f"dims: {dims}")
    print(f"Expected cost 480,340; order 2 x 3; 1 x 2:3; 4 x 5; 1:3 x 4:5")
    print(f"                          or 4 x 5; 2 x 3; 1 x 2:3; 1:3 x 4:5")
    print(f"Got      cost {cost:,}; order {order}")
    print()

    print()
    print(f"Question to hand in the result for:")
    dims = [15123, 12, 5, 15, 32, 15124, 12, 17, 13, 100, 56, 155, 12, 11, 11, 7, 100]
    cost, order = mmul_order(dims)
    print(f"dims: {dims}")
    print(f"Got cost {cost:,}")
    from textwrap import wrap

    print("Got order " + "\n".join(wrap(order, width=72, subsequent_indent=" " * 10)))


if __name__ == "__main__":
    main()
