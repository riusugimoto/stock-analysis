
#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (2023-24)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import array
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    test_and_plot,
    load_dataset,
    load_mnist,
    main,
    handle,
    run,
)
from naive_bayes import NaiveNaiveBayes, NaiveBayes, VQNB
from neural_net import NeuralNetRegressor


def bernoulli_sample(n_sample, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    arr = array.array('i', [])

    for _ in range (n_sample):
        u = rng.random()
        if u < theta:
            x=1
        else:
            x=0
        arr.append(x)
    
    return arr

def bernoulli_sample2(n_sample, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    arr = array.array('i', [])

    for _ in range (n_sample):
        u = rng.random()
        if u < theta:
            x=2
        else:
            x=-5
        arr.append(x)
    
    return arr





@handle("bernoulli-mc")
def bernoulli_mc():
    max_n = 100_000
    theta = 0.26
    n_repeats = 3

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Monte Carlo approximation")
    ax.set_xlim(1, max_n)
    ax.set_ylim(-10, 10)

    for _ in range(n_repeats):

        samples = bernoulli_sample2(max_n, theta)
        # running_mean = np.cumsum(samples) / max_n
        running_mean = np.cumsum(samples) / np.arange(1, max_n + 1)

        ax.plot(running_mean)

    # Showing the analytical expected value
    expected_value = theta * 2 + (1 - theta) * (-5)
    ax.axhline(expected_value, color='red', linestyle='--', label='Expected Value')

    ax.legend()
    plt.show()

    fn = Path("..") / "figs" / "bernoulli_game.pdf"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {fn}")




@handle("gambling")
def gambling():
    starting_amount=20
    goal=200
    bankrupt=0
    theta=0.26
    n_simulations=10000
    success_count = 0

    
    for _ in range(n_simulations):
        balance = starting_amount

        if balance >= goal:
            success_count += 1
            continue

        while 0 < balance < goal:
            if np.random.rand() < theta:
                balance += 2  # Win $2
            else:
                balance -= 5  # Lose $5
            if balance >= goal:
                success_count += 1
                break
            elif balance <= bankrupt:
                break
            
    print(success_count)
    probability = success_count / n_simulations
    return probability


probability = gambling()
print(f"Probability of reaching $200 before going broke: {probability:.5%}")







    

  


################################################################################


def eval_models(models, dataset="mnist", **kwargs):
    X, y, Xtest, ytest = load_dataset(dataset, "X", "y", "Xtest", "ytest", **kwargs)
    for model in models:
        model.fit(X, y)
        yhat = model.predict(Xtest)
        yield np.mean(yhat != ytest)


def eval_model(model, dataset="mnist", **kwargs):
    return next(eval_models([model], dataset=dataset, **kwargs))


@handle("mnist-naive-nb")
def mnist_naive_nb():
    err = eval_model(NaiveNaiveBayes())
    print(f"Test set error: {err:.1%}")


@handle("mnist-nb")
def mnist_nb():
    laps = [1e-9, 0.5, 1, 2.5, 5, 10, 50]
    errs = eval_models([NaiveBayes(prior_alpha=lap, prior_beta=lap) for lap in laps])
    print("NaiveBayes test set errors:")
    for lap, err in zip(laps, errs):
        print(f"  lap    {lap:>4.1f}:  {err:.1%}")


@handle("mnist-logreg")
def mnist_logreg():
    from sklearn.linear_model import LogisticRegression

    # multi_class='multinomial' means softmax loss
    # C is like 1/lambda
    # tol is optimization tolerance
    err = eval_model(
        LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1,
            tol=0.1,
            max_iter=10_000,
            multi_class="multinomial",
        )
    )
    print(f"Test set error: {err:.1%}")


@handle("mnist-vqnb")
def mnist_vqnb():
    ks = [2, 3, 4, 5]
    models = [VQNB(k=k) for k in ks]
    print("VQNB test set errors:")
    for k, err in zip(ks, eval_models(models)):
        print(f"  k = {k}:  {err:.1%}")

    model = models[-1]
    fig, axes = plt.subplots(
        k,
        10,
        figsize=(8, 8 * k / 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for y in range(10):
        for b in range(k):
            ps = np.zeros(784)  # get the probabilities from your model
            axes[b][y].imshow(ps.reshape((28, 28)), "gray")
            axes[b][y].set_axis_off()
    fig.savefig("../figs/vqnb_probs.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Plots in ../figs/vqnb_probs.pdf")

    for y in range(10):  # For each class
        for z in range(k):  # For each cluster within class
            # Extract the probabilities for class y and cluster z
            ps = model.theta[y, z]  # Assuming this is how your model stores probabilities
            # Plot the probabilities as a 28x28 image
            if k > 1:
                ax = axes[z][y]
            else:
                ax = axes[y]  # Adjust for when k = 1 to avoid indexing issues
            ax.imshow(ps.reshape((28, 28)), cmap="gray")
            ax.set_axis_off()
    
    plt.suptitle("Estimates for $p(x_j = 1 | z, y)$")
    fig.savefig("../figs/vqnb_probs.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Plots in ../figs/vqnb_probs.pdf")


################################################################################


@handle("nn-regression")
def nn_regression():
    X, y = load_dataset("basis_data", "X", "y")

    model = NeuralNetRegressor([10], plot_end=True)
    model.fit(X, y)
    yhat = model.predict(X)
    print("Training error: ", np.mean((yhat - y) ** 2))
    print("Figure in ../figs/regression-net.pdf")



@handle("mnist-mlp")
def mnist_mlp():
    import torch_net as tn

    model = tn.TorchNeuralNetClassifier([3], device="cpu")
    # might run faster if you change device= to use your GPU:
    #    "cuda" on Linux/Windows if you have an appropriate GPU and PyTorch install
    #    "mps" if you have a recent Mac
    print(f"mlp test set error: {eval_model(model):.1%}")
    # List all parameters and their shapes for the instantiated model
    params = list(model.parameters())
    param_shapes = [p.shape for p in params]
    print(param_shapes, params)



@handle("mnist-cnn")
def mnist_cnn():
    import torch_net as tn

    model = tn.Convnet(
        device="cpu",
    )
    # might run faster if you change device= to use your GPU:
    #    "mps" if you have a recent Mac
    #    "cuda" on Linux/Windows if you have an appropriate GPU and PyTorch install
    print(f"cnn test set error: {eval_model(model):.1%}")


if __name__ == "__main__":
    main()
