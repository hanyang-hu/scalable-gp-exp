import torch
import gpytorch

import time
from scipy.io import loadmat
from math import floor

from scripts.fast_cache_predict import FastCachePredictor

# assume elevators.mat is in the same directory as this script
data = torch.Tensor(loadmat('./elevators.mat')['data'])
X = data[:, :-1].cuda()
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1].cuda()
start_n = 0 # int(floor(0.4 * len(X)))
train_n = int(floor(0.6 * len(X)))
test_n = int(floor(0.8 * len(X)))
train_x = X[start_n:train_n, :].contiguous()
train_y = y[start_n:train_n].contiguous()

print("Training data size: ", train_x.size(0))

test_x = X[test_n:, :].contiguous()
test_y = y[test_n:].contiguous()

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

output_device = torch.device('cuda:0')

train_x, train_y = train_x.to(output_device), train_y.to(output_device)
test_x, test_y = test_x.to(output_device), test_y.to(output_device)

print(
    f"Num train: {train_y.size(-1)}\n"
    f"Num test: {test_y.size(-1)}"
)
dim = train_x.size(-1)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=dim
            )
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Because we know some properties about this dataset,
# we will initialize the lengthscale to be somewhat small
# This step isn't necessary, but it will help the model converge faster.
model.covar_module.base_kernel.lengthscale = 0.05

import time
import tqdm

start_time = time.time()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

model.covar_module.base_kernel.lengthscale = torch.tensor(
    [
        [
            1.4635, 
            1.4874, 
            1.3254, 
            1.5282, 
            1.6032, 
            1.1936, 
            1.4386, 
            0.3125, 
            1.6088,
            0.9968, 
            0.8852, 
            0.8852, 
            0.7260, 
            1.5136, 
            0.0500, 
            1.6657,
            0.0500, 
            0.7260
        ]
    ]
).cuda()

training_iter = 20
iterator = tqdm.tqdm(range(training_iter), desc="Training")
start_time = time.time()
for i in iterator:
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    print_values = dict(
        loss=loss.item(),
        ls=model.covar_module.base_kernel.lengthscale.norm().item(),
        os=model.covar_module.outputscale.item(),
        noise=model.likelihood.noise.item(),
        mu=model.mean_module.constant.item(),
    )
    iterator.set_postfix(**print_values)
    loss.backward()
    optimizer.step()

end_time = time.time()

print("Training Time for {} iterations: {}\n".format(training_iter, end_time - start_time))

model.eval()
k = 20
testing_iter = 50

train_y = train_y.squeeze(0)

with torch.no_grad():
    predictor = FastCachePredictor(model, max_root_decom_size=k)
    start_time = time.time()
    for i in range(testing_iter):
        res = predictor.predict(test_x)
        std = res[1].sqrt()
        predictor.update(
            X[train_n+i:train_n+i+1], 
            y[train_n+i:train_n+i+1]
        )
    end_time = time.time()

    rmse = torch.mean((res[0] - test_y).pow(2)).sqrt()

print("Testing Time for Fast Cache Update with {} iterations: {}\n".format(testing_iter, end_time - start_time))
print("RMSE: ", rmse.item())

# Time: 0.41591954231262207
# RMSE: 0.09760726243257523

del predictor

# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(k):
#     start_time = time.time()
#     for i in range(testing_iter):
#         res = model(test_x)
#         std = res.variance.sqrt()
#         model = model.get_fantasy_model(
#             X[train_n+i:train_n+i+1], 
#             y[train_n+i:train_n+i+1]
#         )
#     end_time = time.time()

#     rmse = torch.mean((res.mean - test_y).pow(2)).sqrt()

# print("Testing Time for GPyTorch Implementation with {} iterations: {}\n".format(testing_iter, end_time - start_time))
# print("RMSE: ", rmse.item())

# Time: 41.078134059906006
# RMSE: 0.09764832258224487

