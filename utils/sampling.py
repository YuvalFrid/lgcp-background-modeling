from pdb import set_trace as st 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize





class CustomPDFModel:
    #### A handler for generating 'n_samples' samples and fitting toy data to a given PDF 'pdf_func' with 'true_params'.  
    def __init__(self, pdf_func, true_params, n_samples, seed=None):
        self.pdf_func = pdf_func
        self.true_params = true_params
        self.n_samples = n_samples
        self.seed = seed
        self.samples = None
        self.fitted_params = None
        self.covariance = None  # inverse Hessian (cov matrix)
        self.jacobian = None    # gradient at solution

        if seed is not None:
            np.random.seed(seed)

    def normalization_constant(self, params):
        Z, _ = quad(lambda x: self.pdf_func(x, *params), 0, 1)
        return Z

    def normalized_pdf(self, x, params):
        Z = self.normalization_constant(params)
        return self.pdf_func(x, *params) / Z

    def sample(self):
        samples = []
        ys = self.pdf_func(np.linspace(0,1,1000), *self.true_params)
        max_pdf = ys.max()
        while len(samples) < self.n_samples:
            x_candidate = np.random.uniform(0,1)
            y_candidate = np.random.uniform(0, max_pdf)
            if y_candidate <= self.pdf_func(x_candidate, *self.true_params):
                samples.append(x_candidate)
        self.samples = np.array(samples)
        return self.samples

    def neg_log_likelihood(self, params):
        if np.any(np.array(params) < 0):
            return np.inf
        Z = self.normalization_constant(params)
        if Z == 0:
            return np.inf
        pdf_vals = np.array([self.pdf_func(x, *params)/Z for x in self.samples])
        if np.any(pdf_vals <= 0):
            return np.inf
        return -np.sum(np.log(pdf_vals))

    def fit(self, initial_guess):
        if self.samples is None:
            raise RuntimeError("No samples to fit. Run sample() first.")
        bounds = [(0, None)] * len(initial_guess)
        result = minimize(self.neg_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B', jac=None)
        if not result.success:
            raise RuntimeError("MLE fit failed:", result.message)
        self.fitted_params = result.x

        # Save inverse Hessian matrix as covariance matrix if available
        if hasattr(result, 'hess_inv'):
            if callable(result.hess_inv):
                # For L-BFGS-B, hess_inv is a LinearOperator; try to convert to dense matrix if possible
                try:
                    self.covariance = result.hess_inv.todense()
                except Exception:
                    self.covariance = None
            else:
                self.covariance = result.hess_inv
        else:
            self.covariance = None

        # Save jacobian (gradient) at solution if available
        self.jacobian = result.jac if hasattr(result, 'jac') else None

        return self.fitted_params

    def pdf_uncertainty_band(self, x_points, n_sigma=1):
        """
        Calculate uncertainty band on PDF from covariance of fitted parameters.
        Uses simple linear error propagation:

        var(PDF) ≈ J(x) @ Cov @ J(x).T

        where J(x) = gradient of PDF w.r.t parameters at x.
        """

        if self.covariance is None or self.fitted_params is None:
            return None, None

        grad_vecs = []

        # Numerical gradient of normalized_pdf w.r.t parameters at each x
        delta = 1e-6
        for x in x_points:
            grads = []
            for i in range(len(self.fitted_params)):
                params_up = self.fitted_params.copy()
                params_down = self.fitted_params.copy()
                params_up[i] += delta
                params_down[i] -= delta
                f_up = self.normalized_pdf(x, params_up)
                f_down = self.normalized_pdf(x, params_down)
                grad = (f_up - f_down) / (2 * delta)
                grads.append(grad)
            grad_vecs.append(grads)

        grad_vecs = np.array(grad_vecs)  # shape (len(x_points), n_params)

        # Variance at each x: v = g.T @ Cov @ g
        variances = np.einsum('ij,jk,ik->i', grad_vecs, self.covariance, grad_vecs)
        uncertainties = n_sigma * np.sqrt(np.abs(variances))

        return uncertainties

    def plot(self, bins=30, show_uncertainty=True, n_sigma=1):
        x = np.linspace(0,1,500)
        plt.figure(figsize=(8,5))

        y_true = self.normalized_pdf(x, self.true_params)
        plt.plot(x, y_true, label='True PDF',color = 'k',linestyle = '--',lw=3,)

        plt.hist(self.samples, bins=bins, range = (0,1),density=True, alpha=0.5, label='Samples',color = 'blue',histtype = 'step')

        if self.fitted_params is not None:
            y_fit = self.normalized_pdf(x, self.fitted_params)
            plt.plot(x, y_fit, label='Fitted PDF', lw=3,color = 'red', linestyle='-')

            if show_uncertainty and self.covariance is not None:
                unc = self.pdf_uncertainty_band(x, n_sigma=n_sigma)
                if unc is not None:
                    plt.fill_between(x, y_fit - unc, y_fit + unc, color='red', alpha=0.3,
                                     label=f'±{n_sigma}σ uncertainty band')
        
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title('PDF and Sample Histogram')
        plt.show()
        return y_fit, unc
