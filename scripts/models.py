import gpytorch

# We will use the simplest form of GP model, exact inference
class ExactGPRBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        dim = train_x.size(-1)
        super(ExactGPRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=dim,
            )
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

# We will use the simplest form of GP model, exact inference
class ExactGPRBFKeOps(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        dim = train_x.size(-1)
        super(ExactGPRBFKeOps, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(
                ard_num_dims=dim,
            )
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)