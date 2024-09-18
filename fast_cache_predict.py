import torch
import gpytorch
from linear_operator.operators import MatmulLinearOperator

"""
Implementation of fast cache update for "exact" GPs based on LOVE.
Check out the following papers
    1. https://arxiv.org/abs/1803.06058;
    2. https://arxiv.org/abs/2006.15779 
for more details.

For max_root_decom_size=k, the time and space complexities are:
    1. O(nk) time for prediction;
    2. O(nk) time for cache update;
    3. O(nk) space for cache storage.

Limitations:
    1. Each update increase k by 1 (hence recommended to not update too frequently);
    2. Lack theoretical guarantee for accuracy;
    3. Currently only support constant mean and noise (other options can be implemented but not necessary);
    4. Currently only support updating one new data point at a time (other options can be implemented but not necessary).

Note. When training data is small, GPyTorch does not use LOVE for prediction.

TODO: Save and load the model with the cache.
"""
class FastCachePredictor():
    def __init__(self, trained_model, max_root_decom_size=20, precompute=True, probe_vector=None):
        # Extract mean and covariance module
        self.model = trained_model
        self.mu = self.model.mean_module.constant
        self.noise = self.model.likelihood.noise # this is necessary for update
        self.kernel = self.model.covar_module

        # Extract training data from the trained model
        self.train_X = self.model.train_inputs[0]
        self.train_y = self.model.train_targets

        self.k = max_root_decom_size

        # Extract mean and LOVE cache from the trained model
        # From then on, we have nothing to do with GPyTorch
        if precompute:
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(self.k):
                probe_vector = probe_vector if probe_vector is not None else self.train_X[0:1] + torch.randn_like(self.train_X[0:1])
                self.model(probe_vector) # Precomputation with arbitrary input
                self.mean_cache = self.model.prediction_strategy.mean_cache
                self.love_cache = self.model.prediction_strategy.covar_cache

    """
    The default non-lazy option optimizes batched dot product via einsum.

    If lazy=True, the returned var is a linear operator.
    It will not be computed if not necessary (i.e. used in subsequent computations).
    Extract the posterior variance through `var.diag()` if you want to access it directly.
    """
    def predict(self, test_X, return_dist=False, lazy=False):
        # Predict mean
        test_train_covar = self.kernel(test_X, self.train_X) # get train_test_covar
        preds_mean = self.mu + test_train_covar.matmul(self.mean_cache) # get mean prediction from mean cache

        # Predict variance
        test_test_covar = self.kernel(test_X, diag=not lazy)
        covar_inv_quad_form_root = test_train_covar.matmul(self.love_cache)
        if lazy:
            preds_var = test_test_covar + MatmulLinearOperator(
                covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
            )
        else:
            preds_var = test_test_covar - torch.einsum(
                'ij,ij->i', # batched dot product
                covar_inv_quad_form_root,
                covar_inv_quad_form_root
            )

        if return_dist:
            return torch.distributions.normal.Normal(
                loc=preds_mean, 
                scale=preds_var.diag().sqrt().clamp_min(1e-9)
            )
        else:
            return (preds_mean, preds_var)

    """
    Fast cache update in O(nk) time.
    Resulting in a (n+1)-D new mean cache and (n+1)x(k+1) new LOVE cache.
    """
    def update(self, new_train_X, new_train_y):
        u = self.kernel(new_train_X, self.train_X).evaluate() # u = fant_train_covar
        s = (self.kernel(new_train_X).evaluate() + self.noise)
        
        # Compute new mean cache
        L12 = u.matmul(self.love_cache) # to be used in LOVE cache update
        Q = L12.matmul(self.love_cache.transpose(-1, -2)) # Q approx K^{-1}u
        b = new_train_y - self.mu - torch.mul(u, self.mean_cache).sum()
        b = b / (s - torch.mul(Q, u).sum()) # b = (y - mu - u.TK^{-1}y) / (s - u.TK^{-1}u)
        new_mean_cache = torch.hstack(
            [
                self.mean_cache - Q * b, 
                b
            ]
        ).squeeze(0)

        # Compute new LOVE cache
        upper_rect = torch.cat(
            [
                self.love_cache, 
                torch.zeros_like(self.love_cache[0:1, :])
            ]
        )
        L22_inv = 1 / torch.sqrt(s - torch.mul(L12, L12).sum())
        lower_rect = torch.hstack(
            [
                -L22_inv * Q, 
                L22_inv
            ]
        ).transpose(-1, -2)
        new_love_cache = torch.cat(
            [
                upper_rect, 
                lower_rect
            ], 
            dim=-1
        )

        # Update dataset and mean/LOVE cache
        self.train_X = torch.cat([self.train_X, new_train_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_train_y], dim=0)
        self.mean_cache = new_mean_cache
        self.love_cache = new_love_cache


"""
TODO: Batch version of the fast cache update for exact GPs based on LOVE.
"""
class BatchFastCachePredictor():
    pass