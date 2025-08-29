from pdb import set_trace as st 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS


class TorchModel(nn.Module):
    def __init__(self, data_input, length_scale=0.5, var_scale=1.0, optimizing_points=100):
        super().__init__()
        self.data = torch.tensor(data_input.ravel(), dtype=torch.float32)
        self.optimizing_points = optimizing_points
        self.register_parameter('log_length_scale', nn.Parameter(torch.log(torch.tensor(length_scale))))
        self.register_parameter('log_var_scale', nn.Parameter(torch.log(torch.tensor(var_scale))))
        self._x_vals(optimizing_points)
        self.mean = torch.zeros(self.x.size(0), dtype=torch.float32)
        self.samples = torch.randn(10000, optimizing_points)  # fixed standard normal samples

    def _x_vals(self, bins):
        x = torch.linspace(0, 1, bins + 1)
        self.x = 0.5 * (x[1:] + x[:-1])
        self.distmat = (self.x[None, :] - self.x[:, None])
        self.idx()

    def idx(self):
        idx_left = torch.searchsorted(self.x, self.data, right=True) - 1
        idx_left = torch.clamp(idx_left, 0, self.x.size(0) - 2)
        self.idx_left = idx_left
        self.idx_right = idx_left + 1
        self.x_left = self.x[self.idx_left]
        self.x_right = self.x[self.idx_right]

    def linear_interpolation(self, intensity):
        y_left = intensity[:, self.idx_left]
        y_right = intensity[:, self.idx_right]
        weight = (self.data - self.x_left) / (self.x_right - self.x_left)
        y_interp = y_left + (y_right - y_left) * weight
        return y_interp

    def _compute_cov(self):
        l = torch.exp(self.log_length_scale)
        v = torch.exp(self.log_var_scale)
        distsq = (self.distmat / l) ** 2
        cov = v * torch.exp(-0.5 * distsq)
        cov += 1e-5 * torch.eye(self.x.size(0))  # Numerical stability
        return cov

    def log_likelihood(self, intensity):
        N = self.data.size(0)
        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(0)
        exp_int = torch.exp(intensity)
        integral = N * torch.trapezoid(exp_int, self.x, dim=1)
        interpolated = N * torch.exp(self.linear_interpolation(intensity))
        log_term = torch.log(interpolated).sum(dim=1)
        return log_term - integral

    def log_posterior(self, intensity):
        cov = self._compute_cov()
        L = torch.linalg.cholesky(cov)
        invL = torch.cholesky_inverse(L)
        delta = (intensity - self.mean)
        quad_term = (delta @ invL @ delta.T).diagonal()
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        norm_const = 0.5 * (logdet + intensity.size(1) * torch.log(torch.tensor(2 * torch.pi)))
        return self.log_likelihood(intensity) - 0.5 * quad_term - norm_const

    def marginal_log_likelihood(self):
        cov = self._compute_cov()
        L = torch.linalg.cholesky(cov)
        samples = self.samples @ L.T
        log_likes = self.log_likelihood(samples)
        log_max = torch.max(log_likes)
        rel = torch.exp(log_likes - log_max)
        return log_max + torch.log(torch.mean(rel))

    def grad_optimize_hyperparameters(self, epochs=100, lr=0.1):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for step in range(epochs):
            optimizer.zero_grad()
            loss = -self.marginal_log_likelihood()  # NEGATIVE: we want to maximize
            loss.backward()
            optimizer.step()
            if epochs % 10 == 0 or epochs == epochs - 1:
                print(f"[Epoch {step}] Marginal LL: {-loss.item():.4f}, "
                      f"Length: {torch.exp(self.log_length_scale).item():.4f}, "
                      f"Var: {torch.exp(self.log_var_scale).item():.4f}")

    def mean_fit(self,epochs = 100, lr=0.01):
#        mean = self.mean.clone().detach().requires_grad_(True)
        # Step 1: Compute kernel matrix from current hyperparameters
        K = self._compute_cov()  # should return torch.Tensor of shape (N, N)

        # Step 2: Cholesky decomposition
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(K.shape[0], device=K.device))  # for numerical stability

        # Step 3: Sample from GP prior
        epsilon = torch.randn(K.shape[0], device=K.device)
        mean = (L @ epsilon).detach().requires_grad_(True)  # shape (N,)

        optimizer = optim.Adam([mean], lr=lr)
        for i in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = -self.log_posterior(mean.unsqueeze(0))
            loss.backward()
            optimizer.step()
        self.mean = mean.detach()

    def run_nuts_sampler(self, steps=100, warmup=10):
        D = self.mean.shape[0]

        def model():
            theta = pyro.sample("theta", dist.MultivariateNormal(
                torch.zeros(D), self._compute_cov()))
            pyro.factor("log_like", self.log_likelihood(theta.unsqueeze(0)))
    
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=steps, warmup_steps=warmup)
        mcmc.run()
        lower = np.exp(np.array(torch.quantile(mcmc.get_samples()["theta"],0.16,axis = 0)))
        median = np.exp(np.array(torch.quantile(mcmc.get_samples()["theta"],0.5,axis = 0)))
        upper = np.exp(np.array(torch.quantile(mcmc.get_samples()["theta"],0.84,axis = 0)))
        scale = np.trapz(median,self.x)
        median /= scale
        lower /= scale
        upper /= scale
        return lower,median,upper

    def plot(self,true_pdf,ref_curve,ref_unc,MCMC_steps = 1000):
        lower,median,upper = self.run_nuts_sampler(steps = MCMC_steps,warmup = MCMC_steps//2)
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

