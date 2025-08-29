## LGCP Background Modeling

This repository implements a Log Gaussian Cox Process (LGCP) for modeling 1D probability distributions.

It improves upon the previous version described in [our paper](https://arxiv.org/abs/2508.11740), with the following updates:
- The log marginal likelihood is now written in a form that supports gradient-based optimization.
- Hyperparameter optimization uses a fast SGD loop instead of slow MCMC.
- Posterior estimation is updated to use NUTS sampling via the Pyro library.

## Motivation

In many high-energy physics analyses, there's a need to fit 1D data to an unknown probability density function (PDF).
This LGCP method provides a fast, automated fit using a Bayesian, non-parametric approach for background estimation.

The entire implementation is written from scratch using NumPy, PyTorch, and Pyro.

## Setup

    source setup_env.sh

## Activation

    source .venv/bin/activate

## Demo

In the 'notebooks/' directory, see 'BKG_Fit.ipynb' for an interactive demo.
It includes:
- Generating toy data from a known PDF
- Fitting with MLE assuming full knowledge of the function (idealized case)
- Fitting with the LGCP method and comparing the results

## License

MIT License — see LICENSE for details.

