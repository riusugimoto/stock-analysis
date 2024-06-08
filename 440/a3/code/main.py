#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (2023-24)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import pickle

import numpy as np
from scipy.special import betaln
from scipy.optimize import minimize
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar


# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from generative_classifiers import GDA, TDA
from utils import (
    test_and_plot,
    load_dataset,
    main,
    handle,
    run,
)





def bernoulli_nll(theta, n_pos, n_total):
    epsilon = 1e-10
    theta_adjusted = np.clip(theta, epsilon, 1-epsilon)
    n_neg = n_total - n_pos
    return -n_pos * np.log(theta_adjusted) - n_neg * np.log1p(-theta_adjusted)
    # np.log1p(x) == np.log(1 + x)  except that log1p is more accurate for small |x|


def berns_nll(thetas, data):
    return sum(
        bernoulli_nll(theta, n_pos, n_total)
        for theta, (n_pos, n_total) in zip(thetas, data)
    )


@handle("eb-base")
def eb_base():
    n, n_test = load_dataset("cancer_data", "n", "n_test")
    n_groups = n.shape[0]

    theta = np.full(n_groups, 0.5)
    print(f"NLL for theta = 0.5: {berns_nll(theta, n_test) : .1f}")

    mle_thetas = n[:, 0] / n[:, 1]
    with np.errstate(all="ignore"):  # ignore numerical warnings about nans
        print(f"NLL for theta = MLE: {berns_nll(mle_thetas, n_test) : .1f}")


@handle("eb-map")
def eb_map():
    n, n_test = load_dataset("cancer_data", "n", "n_test")
    alpha, beta = 2, 2
    map_thetas = (n[:, 0] + alpha - 1) / (n[:, 1] + alpha + beta - 2)
    test_nll = berns_nll(map_thetas, n_test)
    
    with np.errstate(all="ignore"):  # ignore numerical warnings about nans
        print(f"Test NLL with Laplace smoothing: {test_nll : .1f}")


def marginal_likelihood(params, n, n_test):
    a, b = params
    n1 = n[:, 0].sum()  # Total number of positives in the training set
    n0 = n[:, 1].sum() - n1  # Total number of negatives in the training set
    return -(betaln(a + n1, b + n0) - betaln(a, b))


@handle("eb-bayes")
def eb_bayes():
    n, n_test = load_dataset("cancer_data", "n", "n_test")
    
    # The prior hyperparameters
    alpha_prior = 2
    beta_prior = 2
    
    total_successes = n[:, 0]
    total_trials = n[:, 1]
    
    #  posterior parameters
    alpha_post = alpha_prior + n[:, 0]
    beta_post = beta_prior + n[:, 1] - n[:, 0]
    
    # The posterior predictive distribution for a new observation
    theta_post_pred = alpha_post / (alpha_post + beta_post)
    
    
    nll_post_pred = berns_nll(theta_post_pred, n_test)
    # Max log marginal likelihood using posterior parameters
    max_log_marginal_likelihood = betaln(alpha_post, beta_post) - betaln(alpha_prior, beta_prior)
    
    print(nll_post_pred)

    return theta_post_pred, nll_post_pred, max_log_marginal_likelihood

# Now you would include this function in your main handling code or call it directly as needed.


@handle("eb-pooled")
def eb_pooled():
    n, n_test = load_dataset("cancer_data", "n", "n_test")

    # Pooling data
    total_pos = n[:, 0].sum()
    total = n[:, 1].sum()

    # MLE for pooled data
    theta_mle = total_pos / total
    nll_mle = berns_nll([theta_mle] * len(n_test), n_test)

    # MAP for pooled data (using example alpha, beta = 2, 2 for simplicity)
    alpha, beta = 2, 2
    theta_map = (total_pos + alpha - 1) / (total + alpha + beta - 2)
    nll_map = berns_nll([theta_map] * len(n_test), n_test)

    # Posterior Predictive for pooled data
    alpha = 2
    beta = 2
    
    alpha_post = alpha + total_pos
    beta_post = beta + total - total_pos
    theta_post_pred = alpha_post / (alpha_post + beta_post)
    nll_post_pred = berns_nll([theta_post_pred] * len(n_test), n_test)

    print(f"Pooled NLL (MLE): {nll_mle:.2f}")
    print(f"Pooled NLL (MAP): {nll_map:.2f}")
    print(f"Pooled NLL (Posterior Predictive): {nll_post_pred:.2f}")




# Function to compute log marginal likelihood given m and k
def log_marginal_likelihood(alpha_beta, n1, n0):
    alpha, beta = alpha_beta
    return betaln(alpha + n1, beta + n0) - betaln(alpha, beta)


@handle("eb-max")
def eb_max():
    n, n_test = load_dataset("cancer_data", "n", "n_test")

    total_pos = n[:, 0].sum()
    total = n[:, 1].sum()
    # Set m to the MLE of theta
    m = total_pos / total

    k_values = np.linspace(1, 10000, 200)  
    
    def log_marginal_likelihood(alpha, beta, n1, n0):
        return betaln(alpha + n1, beta + n0) - betaln(alpha, beta)
    
    max_log_marginal = -np.inf
    best_k = None
    for k in k_values:
        alpha = m * k
        beta = (1 - m) * k
        current_log_marginal = log_marginal_likelihood(alpha, beta, total_pos, total-total_pos)
        if current_log_marginal > max_log_marginal:
            max_log_marginal = current_log_marginal
            best_k = k

    # Calculate the optimal alpha and beta
    optimal_alpha = m * best_k
    optimal_beta = (1 - m) * best_k

    print(f"Optimal alpha: {optimal_alpha}")
    print(f"Optimal beta: {optimal_beta}")
    print(f"Maximum Log Marginal Likelihood: {max_log_marginal}")




# Now call this function in your main handling code or as needed




def objective_function(params, n):
    alpha, beta = params
    #to m, k
    m = alpha / (alpha + beta)
    k = alpha + beta
    
    #  log prior for m and k
    log_prior_m = (0.01 - 1) * np.log(m) + (9.9 - 1) * np.log(1 - m)
    log_prior_k = -2 * np.log(1 + k)
    
    # Calculate the log marginal likelihood
    n1 = n[:, 0].sum() 
    n0 = n[:, 1].sum() - n1 
    log_marginal_likelihood = -(betaln(alpha + n1, beta + n0) - betaln(alpha, beta))
    
    # Combine log priors and log marginal likelihood
    return -(log_prior_m + log_prior_k + log_marginal_likelihood)




@handle("eb-newprior")
def eb_newprior():
    n, n_test = load_dataset("cancer_data", "n", "n_test")
    # Brute force search over the range {0.1, 0.2, ..., 9.9} for alpha and beta

    alpha_beta_range = np.arange(0.1, 10, 0.1)
    best_objective_value = np.inf
    best_params = None

    for alpha in alpha_beta_range:
        for beta in alpha_beta_range:
            current_objective_value = objective_function((alpha, beta), n)
            if current_objective_value < best_objective_value:
                best_objective_value = current_objective_value
                best_params = (alpha, beta)

    best_params, -best_objective_value 
    
    print (best_params, -best_objective_value)
    




def objective_function_sep(params, n):
    alpha, beta = params

    m = alpha / (alpha + beta)
    k = alpha + beta

    log_prior_m = (0.01 - 1) * np.log(m) + (9.9 - 1) * np.log(1 - m)
    log_prior_k = -2 * np.log(1 + k)
    
    # Sum of log marginal likelihoods across all groups
    log_marginal_likelihoods = np.sum(betaln(n[:,0] + alpha, n[:,1] - n[:,0] + beta) - betaln(alpha, beta))
    
    return -(log_prior_m + log_prior_k + log_marginal_likelihoods)


@handle("eb-newprior-sep")
def eb_newprior_sep():
    n, n_test = load_dataset("cancer_data", "n", "n_test")

    # Brute force search over the range {0.1, 0.2, ..., 9.9} for alpha and beta

    alpha_beta_range = np.arange(0.1, 10, 0.1)
    best_objective_value = np.inf
    best_params = None

    for alpha in alpha_beta_range:
        for beta in alpha_beta_range:
            current_objective_value = objective_function_sep((alpha, beta), n)
            if current_objective_value < best_objective_value:
                best_objective_value = current_objective_value
                best_params = (alpha, beta)

    best_alpha, best_beta = best_params
    
    print (best_params, -best_objective_value)

      
    posterior_predictive_probs = (n_test[:,0] + best_alpha) / (n_test[:,1] + best_alpha + best_beta)

    nll_test_data = -np.sum(n_test[:,0] * np.log(posterior_predictive_probs)
                             + (n_test[:,1] - n_test[:,0]) * np.log(1 - posterior_predictive_probs))


    best_params, best_objective_value, nll_test_data

    print(best_params, best_objective_value, nll_test_data)

    alpha_optimized = 4.8
    beta_optimized = 9.9
    # Calculate the posterior predictive probability of seeing a 1 for a new group
    posterior_predictive_probability_1 = alpha_optimized / (alpha_optimized + beta_optimized)

    print(posterior_predictive_probability_1)

################################################################################





def eval_models(models, ds_name):
    X, y, X_test, y_test = load_dataset(ds_name, "X", "y", "X_test", "y_test")
    for model in models:
        model.fit(X, y)
        y_hat = model.predict(X_test)
        yield np.mean(y_hat != y_test)


def eval_model(model, ds_name):
    return next(eval_models([model], ds_name))


@handle("gda")
def gda():
    model = KNeighborsClassifier(n_neighbors=1)
    print(f"{model.n_neighbors}-NN test error: {eval_model(model, 'gauss_noise'):.1%}")

    model = GDA()
    print(f"GDA  test error: {eval_model(model, 'gauss_noise'):.1%}")


@handle("tda")
def tda():
    model = TDA()
    print(f"TDA  test error: {eval_model(model, 'gauss_noise'):.1%}")


if __name__ == "__main__":
    main()
