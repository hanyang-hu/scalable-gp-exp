from uci_datasets import Dataset
import gpytorch
import torch

from scripts import subsampling

subsample_methods = {
    'random_sampling': subsampling.random_sampling,
    'FPS': subsampling.farthest_point_sampling,
    'anchor_net': subsampling.anchor_net_method
}


# We will use the simplest form of GP model, exact inference
# We choose Matern kernel with nu=5/2
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        dim = train_x.size(-1)
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=dim,
                nu=2.5
            )
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

if __name__ == '__main__':
    import argparse
    import tqdm
    import time

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', '-d', type=str, default='elevators', help="Name of the dataset, see https://github.com/treforevans/uci_datasets/blob/master/README.md for more details.")
    argparser.add_argument('--fraction', '-f', type=float, default=0.1, help="Proportion of the dataset to subsample.")
    argparser.add_argument('--split', type=int, default=0, choices=range(0, 10), help="10-fold cross validation split of the dataset.")
    argparser.add_argument('--seed', type=int, default=42, help="Random seed, default to be 42.")
    argparser.add_argument('--method', '-m', type=str, default='FPS', choices=['random_sampling', 'FPS', 'anchor_net'], help="Subsampling method to use.")
    argparser.add_argument('--iter', type=int, default=100, help="Number of iterations for training the GP model.")

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Dataset(args.dataset)
    X_train, y_train, X_test, y_test = data.get_split(split=args.split)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device).squeeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device).squeeze(-1)

    # subsample the training data
    subsample_method = subsample_methods[args.method]
    if args.fraction == 1.0:
        subsampled_idx = torch.arange(len(X_train))
    else:
        subsampled_idx = subsample_method(X_train, int(len(X_train) * args.fraction))

    # standardize the data
    subsampled_X = X_train[subsampled_idx].to(device)
    subsampled_y = y_train[subsampled_idx].to(device)
    mean_X = subsampled_X.mean(dim=0)
    std_X = subsampled_X.std(dim=0) + 1e-7
    subsampled_X = (subsampled_X - mean_X) / std_X
    mean_y = subsampled_y.mean()
    std_y = subsampled_y.std() + 1e-7
    subsampled_y = (subsampled_y - mean_y) / std_y

    X_test = X_test.to(device)
    X_test = (X_test - mean_X) / std_X
    y_test = y_test.to(device)
    y_test = (y_test - mean_y) / std_y

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(subsampled_X, subsampled_y, likelihood).cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = args.iter
    iterator = tqdm.tqdm(range(training_iter), desc="Training")
    start_time = time.time()
    for i in iterator:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(subsampled_X)
        # Calc loss and backprop gradients
        loss = -mll(output, subsampled_y)
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

    model.eval()

    # test the model, compute RMSE and NLL
    with torch.no_grad():
        preds = model.likelihood(model(X_test))
        rmse = torch.sqrt(torch.mean((preds.mean - y_test) ** 2)).item()
        nll = -torch.distributions.Normal(preds.mean, preds.stddev).log_prob(y_test).mean().item()

    # save the results to a folder "./results/"
    import os

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    with open(f'./results/{args.dataset}_{args.method}_{args.fraction}_{args.split}_{args.seed}.txt', 'w') as f:
        f.write(f'RMSE: {rmse}\n')
        f.write(f'NLL: {nll}\n')