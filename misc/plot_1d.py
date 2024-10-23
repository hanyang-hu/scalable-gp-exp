import gpytorch
import torch
import tqdm
import matplotlib.pyplot as plt

from scripts.fast_cache_predict import FastCachePredictor

f = lambda x : x/5 * torch.sin(x/5)

torch.manual_seed(41)

t = 20

train_x = torch.linspace(0, 40, 4).cuda().unsqueeze(-1)
train_x = torch.cat([train_x + torch.randn_like(train_x) * 0.5 for _ in range(1250)], dim=0)
train_y = (f(train_x) + torch.randn_like(train_x) * 0.1).squeeze(-1)

new_train_x = torch.linspace(40, 100, 10).cuda().unsqueeze(-1)[-4:-1]
new_train_x = torch.cat([new_train_x + torch.randn_like(new_train_x) * 0.5 for _ in range(t)], dim=0)
new_train_x, _ = torch.sort(new_train_x, dim=0)
new_train_y = (f(new_train_x) + torch.randn_like(new_train_x) * 0.1).squeeze(-1)

new_train_x_list = []
new_train_y_list = []

# Splite new_train data into 4 parts
for i in range(3):
    new_train_x_list.append(new_train_x[i*t:(i+1)*t])
    new_train_y_list.append(new_train_y[i*t:(i+1)*t])

test_x = torch.linspace(0, 100, 10000).cuda()

# Initialize the model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        dim = train_x.size(-1)
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
    

likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Train the model
model.train()
likelihood.train()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

model.covar_module.base_kernel.lengthscale = torch.tensor([6.0]).cuda()

training_iter = 5
iterator = tqdm.tqdm(range(training_iter), desc="Training")
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

print(model.covar_module.base_kernel.lengthscale)

model.eval()

import time

mean_manual_list_large = []
std_manual_list_large = []

mean_manual_list = []
std_manual_list = []


# Initialize the fast cache predictor
with torch.no_grad():
    predictor = FastCachePredictor(model, max_root_decom_size=2)
    # print(predictor.mean_cache.shape)
    # print(predictor.love_cache.shape)

    lazy = False

    start_time = time.time()

    res = predictor.predict(test_x, lazy=lazy)
    if lazy:
        var = res[1].diag()
    else:
        var = res[1]
    std = var.sqrt()
    mean_manual_list.append(res[0].cpu().numpy())
    std_manual_list.append(std.cpu().numpy())

    for i in range(3):
        for j in range(new_train_x_list[i].shape[0]):
            predictor.update(new_train_x_list[i][j:j+1], new_train_y_list[i][j:j+1])
        res = predictor.predict(test_x, lazy=lazy)
        if lazy:
            var = res[1].diag()
        else:
            var = res[1]
        std = var.sqrt()
        mean_manual_list.append(res[0].cpu().numpy())
        std_manual_list.append(std.cpu().numpy())

    end_time = time.time()

    print("Updating Time of FastCachePredictor: ", end_time - start_time)

with torch.no_grad():
    model._clear_cache()
    predictor = FastCachePredictor(model, max_root_decom_size=10)
    # print(predictor.mean_cache.shape)
    # print(predictor.love_cache.shape)

    lazy = False

    start_time = time.time()

    res = predictor.predict(test_x, lazy=lazy)
    if lazy:
        var = res[1].diag()
    else:
        var = res[1]
    std = var.sqrt()
    mean_manual_list_large.append(res[0].cpu().numpy())
    std_manual_list_large.append(std.cpu().numpy())

    for i in range(3):
        for j in range(new_train_x_list[i].shape[0]):
            predictor.update(new_train_x_list[i][j:j+1], new_train_y_list[i][j:j+1])
        res = predictor.predict(test_x, lazy=lazy)
        if lazy:
            var = res[1].diag()
        else:
            var = res[1]
        std = var.sqrt()
        mean_manual_list_large.append(res[0].cpu().numpy())
        std_manual_list_large.append(std.cpu().numpy())

    end_time = time.time()

    print("Updating Time of FastCachePredictor: ", end_time - start_time)

mean_exact_gp_list = []
std_exact_gp_list = []

with torch.no_grad():
    start_time = time.time()

    res = model(test_x)
    mean_exact_gp_list.append(res.mean.cpu().numpy())
    std_exact_gp_list.append(res.variance.sqrt().cpu().numpy())
    
    for i in range(3):
        model = model.get_fantasy_model(new_train_x_list[i], new_train_y_list[i])
        res = model(test_x)
        mean_exact_gp_list.append(res.mean.cpu().numpy())
        std_exact_gp_list.append(res.variance.sqrt().cpu().numpy())

    end_time = time.time()
    print("Updating Time of Exact GP: ", end_time - start_time)
    mean_gpy = res.mean
    stddev_gpy = res.variance.sqrt()

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
handles = []
labels = []

for i in range(4):
    axs[1][i].set_ylim(-19, 15)

    # Ground Truth Function
    line_gt, = axs[1][i].plot(test_x.cpu().numpy(), f(test_x).cpu().numpy(),
                           label="Ground Truth Function", linestyle=":", color="green")
    if i==3:
        handles.append(line_gt)
        labels.append("Ground Truth Function")

    # Fast Cache Predictive Mean
    line_manual, = axs[1][i].plot(test_x.cpu().numpy(), mean_manual_list[i],
                                label="Fast Cache Predictive Mean")
    if i==3:
        handles.append(line_manual)
        labels.append("Fast Cache Predictive Mean")

    # Fast Cache Predictive Variance
    fill_manual = axs[1][i].fill_between(
        test_x.cpu().numpy(),
        mean_manual_list[i] - 2 * std_manual_list[i],
        mean_manual_list[i] + 2 * std_manual_list[i],
        alpha=0.5,
        color="green",
        edgecolor='black'
    )
    if i==3:
        handles.append(fill_manual)  # Fill doesn't have a label, handled later
        labels.append("Fast Cache Predictive Variance")

    # GPyTorch Predictive Mean
    line_exact, = axs[1][i].plot(test_x.cpu().numpy(), mean_exact_gp_list[i],
                               label="GPyTorch Predictive Mean", linestyle="--", color="red")
    if i == 3:
        handles.append(line_exact)
        labels.append("GPyTorch Predictive Mean")

    # GPyTorch Predictive Variance
    fill_exact = axs[1][i].fill_between(
        test_x.cpu().numpy(),
        mean_exact_gp_list[i] - 2 * std_exact_gp_list[i],
        mean_exact_gp_list[i] + 2 * std_exact_gp_list[i],
        alpha=0.5,
        color="red",
        edgecolor='black'
    )
    if i == 3:
        handles.append(fill_exact)  # Fill doesn't have a label, handled later
        labels.append("GPyTorch Predictive Variance")

    # Training Points
    scatter_train = axs[1][i].scatter(train_x.cpu().numpy(), train_y.cpu().numpy(),
                                     label="Original Training Points")
    if i==3:
        handles.append(scatter_train)
        labels.append("Original Training Points")

    # Updated Training Points (only for i > 0)
    if i > 0:
        scatter_update = axs[1][i].scatter(new_train_x[0:(i) * t].cpu(),
                                         new_train_y[0:(i) * t].cpu(), label="Updated Training Points")
        if i==3:
            handles.append(scatter_update)
            labels.append("Updated Training Points")

for i in range(4):
    axs[0][i].set_ylim(-19, 15)

    # Ground Truth Function
    line_gt, = axs[0][i].plot(test_x.cpu().numpy(), f(test_x).cpu().numpy(),
                           label="Ground Truth Function", linestyle=":", color="green")
    if i==3:
        handles.append(line_gt)
        labels.append("Ground Truth Function")

    # Fast Cache Predictive Mean (Large)
    line_manual_large, = axs[0][i].plot(test_x.cpu().numpy(), mean_manual_list_large[i],
                                label="Fast Cache Predictive Mean (Large)")
    if i==3:
        handles.append(line_manual_large)
        labels.append("Fast Cache Predictive Mean (Large)")

    # Fast Cache Predictive Variance (Large)
    fill_manual_large = axs[0][i].fill_between(
        test_x.cpu().numpy(),
        mean_manual_list_large[i] - 2 * std_manual_list_large[i],
        mean_manual_list_large[i] + 2 * std_manual_list_large[i],
        alpha=0.5,
        color="green",
        edgecolor='black'
    )
    if i==3:
        handles.append(fill_manual_large)  # Fill doesn't have a label, handled later
        labels.append("Fast Cache Predictive Variance (Large)")

    # GPyTorch Predictive Mean
    line_exact, = axs[0][i].plot(test_x.cpu().numpy(), mean_exact_gp_list[i],
                               label="GPyTorch Predictive Mean", linestyle="--", color="red")
    if i == 3:
        handles.append(line_exact)
        labels.append("GPyTorch Predictive Mean")

    # GPyTorch Predictive Variance
    fill_exact = axs[0][i].fill_between(
        test_x.cpu().numpy(),
        mean_exact_gp_list[i] - 2 * std_exact_gp_list[i],
        mean_exact_gp_list[i] + 2 * std_exact_gp_list[i],
        alpha=0.5,
        color="red",
        edgecolor='black'
    )
    if i == 3:
        handles.append(fill_exact)  # Fill doesn't have a label, handled later
        labels.append("GPyTorch Predictive Variance")

    # Training Points
    scatter_train = axs[0][i].scatter(train_x.cpu().numpy(), train_y.cpu().numpy(),
                                     label="Training Points")
    if i==3:
        handles.append(scatter_train)
        labels.append("Training Points")

    # Updated Training Points (only for i > 0)
    if i > 0:
        scatter_update = axs[0][i].scatter(new_train_x[0:(i) * t].cpu(),
                                         new_train_y[0:(i) * t].cpu(), label="Updated Training Points")
        if i==3:
            handles.append(scatter_update)
            labels.append("Updated Training Points")

# Order the legend items explicitly
ordered_handles = [handles[0],  # Ground Truth Function
                   handles[1],  # Fast Cache Predictive Mean
                   handles[2],  # Fast Cache Predictive Variance
                   handles[3],  # GPyTorch Predictive Mean
                   handles[4],  # GPyTorch Predictive Variance
                   handles[5],  # Training Points
                   handles[6]]  # Updated Training Points

ordered_labels = [labels[0], 
                  labels[1], 
                  labels[2], 
                  labels[3], 
                  labels[4], 
                  labels[5], 
                  labels[6]]

# Create a common legend beneath the subplots in the specified order
fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=4, fontsize='xx-large')


# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.10, 1, 0.95])  # Leave space for the legend

# Show the plot
plt.show()

# plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), label="Predictive mean")
# plt.fill_between(
#     test_x.cpu().numpy(),
#     mean.cpu().numpy() - 2 * stddev.cpu().numpy(),
#     mean.cpu().numpy() + 2 * stddev.cpu().numpy(),
#     alpha=0.5,
#     color="blue",
#     label="Predictive variance",
# )

# plt.plot(test_x.cpu().numpy(), mean_gpy.cpu().numpy(), label="Predictive mean GPyTorch", linestyle="--")
# plt.fill_between(
#     test_x.cpu().numpy(),
#     mean_gpy.cpu().numpy() - 2 * stddev_gpy.cpu().numpy(),
#     mean_gpy.cpu().numpy() + 2 * stddev_gpy.cpu().numpy(),
#     alpha=0.5,
#     color="orange",
#     label="Predictive variance GPyTorch",
# )

# plt.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), label="Training data")
# plt.scatter(new_train_x.cpu().numpy(), new_train_y.cpu().numpy(), label="New training data")

