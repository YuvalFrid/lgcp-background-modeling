import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as st
from scipy.linalg import inv
from scipy.stats import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
import torch

class Model():
    def __init__(self,data_input,length_scale = 0.5,var_scale = 1,optimizing_points = 100):
        self.length_scale = length_scale ### The length scale for the kernel
        self.var_scale = var_scale ### the scaling for the kernel
        self.data = data_input.ravel() ### The evidence input
        self._x_vals(optimizing_points)
        #self.marginal_samples = self.MN.rvs(10000)
        self.marginal_samples = np.random.randn(10000, optimizing_points)  # Shape: (N, D)
    def _x_vals(self,bins):
        ### Based on the amount of required bins, sets up the x binning, RBF, and x values for linear interpolation
        x = np.linspace(0,1,bins+1)
        self.x = 0.5*(x[1:]+x[:-1])
        self.distmat = np.subtract.outer(self.x,self.x)
        self.mean = np.zeros(self.x.size)
        self._update_mat()
        self.idx()

    def idx(self):
        ### indices used for linear interpolation
        idx_left = (np.searchsorted(self.x, self.data, side='right') - 1).ravel()
        idx_left[idx_left < 0] = 0
        idx_left[idx_left == self.x.size - 1] -= 1
        self.idx_left = idx_left
        self.idx_right = self.idx_left + 1
        self.x_left, self.x_right = self.x[self.idx_left], self.x[self.idx_right]

    def linear_interpolation(self,intensity):
        ### linear interpolation, used in log likelihood calculation
        y_left, y_right = intensity[:,self.idx_left], intensity[:,self.idx_right]
        y_interp = y_left + (y_right - y_left) * ((self.data - self.x_left) / (self.x_right - self.x_left))
        return y_interp


    def _update_mat(self):
        ### updates the RBF kernel, and the multivariate GP
        self.covmat = self.var_scale*(np.exp(-0.5*(self.distmat/self.length_scale)**2) + 1e-6*np.eye(self.mean.size))
        self.MN = multivariate_normal(self.mean.ravel(),self.covmat)

    def log_likelihood(self,intensity):
        ### claculates log likelihood
        if len(intensity.shape) == 1:
            intensity.resize(1,intensity.size)
        integral = self.data.size*np.trapz(np.exp(intensity),self.x,axis = 1)
        intensity_values = self.data.size*np.exp(self.linear_interpolation(intensity))
        logs = np.log(intensity_values).sum(axis = 1) - integral
        logs[np.isnan(logs)] = -np.inf
        return logs

    def log_posterior(self,intensity):
        return self.log_likelihood(intensity)+self.MN.logpdf(intensity)

    def marginal_log_likelihood(self,particles = 10000):
#        samples = self.MN.rvs(particles)
        self._update_mat()
        L = np.linalg.cholesky(self.covmat + 1e-3 * np.eye(self.covmat.shape[0]))  # Cholesky decomposition
        samples = self.marginal_samples @ L.T
        log_likes = self.log_likelihood(samples)
        log_max = log_likes.max()
        likes = np.exp(log_likes - log_max)
        return log_max + np.log(likes.mean())

    def grad_optimize_hyperparameters(self, steps=100, lr=0.1,epsilon = 0.001):
        for step in range(steps):
            current_marginal = self.marginal_log_likelihood()
            self.length_scale += epsilon
            length_marginal_p = self.marginal_log_likelihood()
            self.length_scale -= 2*epsilon
            length_marginal_m = self.marginal_log_likelihood()
            self.length_scale += epsilon

            dl = (length_marginal_p-length_marginal_m)/2*epsilon

            self.var_scale += epsilon
            var_marginal_p = self.marginal_log_likelihood()
            self.var_scale -= 2*epsilon
            var_marginal_m = self.marginal_log_likelihood()
            self.var_scale += epsilon

            dv = (var_marginal_p-var_marginal_m)/2*epsilon

            scale = lr/(np.sqrt(dl**2+dv**2))

            self.length_scale += dl*scale
            self.var_scale += dv*scale
            if step % 10 == 0 or step == steps - 1:
                print(f"Step {step}: Marginal = {current_marginal:.4f}, "
                      f"length = {self.length_scale:.4f}, var = {self.var_scale:.4f}")

    def grad_fit(self,lr = 0.01,epochs = 100,epsilon = 0.00001):
        self.mean_list = []
        self._update_mat()
        for i in tqdm(range(epochs)):
            current_posterior = self.log_posterior(self.mean)
            new_vector = self.MN.rvs(1)
            new_vector /= np.sqrt((new_vector**2).sum())
            new_posterior = self.log_posterior(self.mean+epsilon*new_vector)
            grad = (new_posterior - current_posterior)/epsilon
            self.mean += lr*grad*new_vector
            self.mean_list.append(self.mean.copy())
        self._update_mat()



    def plot_median(self,true_pdf,ref_curve,ref_unc, samples=10000):
        vectors = self.MN.rvs(samples)                  # (samples, D)
        log_posteriors = self.log_posterior(vectors)    # (samples,)
        weights = np.exp(log_posteriors - np.max(log_posteriors))  # prevent overflow
        weights /= np.sum(weights)                      # Normalize to sum to 1

        percentiles = [16, 50, 84]
        D = vectors.shape[1]
        results = np.zeros((3, D))  # [16, 50, 84] x D

        for d in range(D):
            sorted_indices = np.argsort(vectors[:, d])
            sorted_vals = vectors[sorted_indices, d]
            sorted_weights = weights[sorted_indices]
            cum_weights = np.cumsum(sorted_weights)

            for i, p in enumerate(percentiles):
                cutoff = p / 100.0
                idx = np.searchsorted(cum_weights, cutoff)
                if idx >= len(sorted_vals):
                    idx = -1
                results[i, d] = sorted_vals[idx]

            # Plot
        median = results[1]
        lower = results[0]
        upper = results[2]
        median = np.exp(median)
        scale = np.trapz(median,self.x)
        median /= scale
        lower = np.exp(lower)
        lower /= scale
        upper = np.exp(upper)
        upper/= scale

        plt.title("LGCP_Fit")
        plt.grid(True)


        true_pdf /= np.trapz(true_pdf,self.x)
        plt.plot(self.x,true_pdf, color = 'black',label = 'True PDF')
        plt.hist(self.data,histtype = 'step',density = True,color = 'blue',range = (0,1),bins = 20)
        plt.plot(self.x, median, label="LGCP Fit", color="crimson")
        ref_x = np.linspace(0,1,ref_curve.size + 1)
        ref_x = 0.5*(ref_x[1:]+ref_x[:-1])
        plt.fill_between(self.x, lower, upper, color="crimson", alpha=0.5, label="LGCP STD")
        plt.plot(ref_x,ref_curve,label = 'Reference',color = 'green')
        plt.fill_between(ref_x, ref_curve - ref_unc,ref_curve+ref_unc, color = 'green',alpha = 0.5,label = 'Reference STD')
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.legend()
    
