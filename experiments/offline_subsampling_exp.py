from uci_datasets import Dataset
import gpytorch
import torch

from scripts import subsampling, models

subsample_methods = {
    'random_sampling': subsampling.random_sampling,
    'FPS': subsampling.farthest_point_sampling,
    'FPS_keops': subsampling.farthest_point_sampling_keops,
    'anchor_net': subsampling.anchor_net_method
}
    

if __name__ == '__main__':
    import argparse
    import tqdm

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', '-d', type=str, default='elevators', help="Name of the dataset, see https://github.com/treforevans/uci_datasets/blob/master/README.md for more details.")
    argparser.add_argument('--fraction', '-f', type=float, default=0.1, help="Proportion of the dataset to subsample.")
    argparser.add_argument('--split', type=int, default=0, choices=range(0, 10), help="10-fold cross validation split of the dataset.")
    argparser.add_argument('--seed', type=int, default=42, help="Random seed, default to be 42.")
    argparser.add_argument('--method', '-m', type=str, default='FPS', choices=subsample_methods.keys(), help="Subsampling method to use.")
    argparser.add_argument('--iter', type=int, default=100, help="Number of iterations for training the GP model.")
    argparser.add_argument('--keops', type=bool, default=False, help="Use KeOps for GP model.")
    argparser.add_argument('--ast', type=bool, default=False, help="Use FPS to first learn a subsampled model, then re-weight the points for subsampling.")

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Dataset(args.dataset)
    X_train, y_train, X_test, y_test = data.get_split(split=args.split)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # standardize the data
    mean_X = X_train.mean(dim=0)
    std_X = X_train.std(dim=0) + 1e-7
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    mean_y = y_train.mean()
    std_y = y_train.std() + 1e-7
    y_train = (y_train - mean_y) / std_y
    y_test = (y_test - mean_y) / std_y

    if args.ast:
        subsample_method = subsample_methods['FPS_keops']
        if args.fraction == 1.0:
            subsampled_idx = torch.arange(len(X_train))
        else:
            subsampled_idx = subsample_method(X_train, int(len(X_train) * args.fraction))
        subsampled_X = X_train[subsampled_idx].to(device)
        subsampled_y = y_train[subsampled_idx].to(device)

         # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        if args.keops:
            model = models.ExactGPRBFKeOps(subsampled_X, subsampled_y, likelihood).cuda()
        else:
            model = models.ExactGPRBF(subsampled_X, subsampled_y, likelihood).cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        training_iter = args.iter
        iterator = tqdm.tqdm(range(training_iter), desc="Pre-Training")
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

        lengthscale = model.covar_module.base_kernel.lengthscale.detach().clone()

        del model

        # re-scale training data using the learned lengthscale
        lengthscale = lengthscale / lengthscale.prod() ** (1 / len(lengthscale))
        X_train_scaled = X_train * lengthscale

        # subsample the training data
        subsampled_idx = subsample_method(X_train_scaled, int(len(X_train) * args.fraction))
        subsampled_X = X_train[subsampled_idx].to(device)
        subsampled_y = y_train[subsampled_idx].to(device)

    else:
        # subsample the training data
        subsample_method = subsample_methods[args.method]
        if args.fraction == 1.0:
            subsampled_idx = torch.arange(len(X_train))
        else:
            subsampled_idx = subsample_method(X_train, int(len(X_train) * args.fraction))

        subsampled_X = X_train[subsampled_idx].to(device)
        subsampled_y = y_train[subsampled_idx].to(device)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    if args.keops:
        model = models.ExactGPRBFKeOps(subsampled_X, subsampled_y, likelihood).cuda()
    else:
        model = models.ExactGPRBF(subsampled_X, subsampled_y, likelihood).cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = args.iter
    iterator = tqdm.tqdm(range(training_iter), desc="Training")
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
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model.likelihood(model(X_test))
        rmse = torch.sqrt(torch.mean((preds.mean - y_test) ** 2)).item()
        nll = -torch.distributions.Normal(preds.mean, preds.stddev).log_prob(y_test).mean().item()

    # save the results to "./results/ooffline_subsampling_exp.json"
    # if the same experiment is run multiple times, replace the file with the new results
    import json
    import os

    if not os.path.exists('results'):
        os.makedirs('results')

    results = {
        'dataset': args.dataset,
        'split': args.split,
        'fraction': args.fraction,
        'method': args.method,
        'seed': args.seed,
        'rmse': rmse,
        'nll': nll, 
        'ast': args.ast,
    }

    results_file = 'results/offline_subsampling_exp.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    # check if the same experiment has been run before
    same_exp = False
    for i in range(len(all_results)):
        if all([all_results[i][k] == results[k] for k in ["dataset", "split", "fraction", "method", "seed", "ast"]]):
            all_results[i] = results
            same_exp = True
            break
    if not same_exp:
        all_results.append(results)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_file}")
    print(json.dumps(results, indent=4))