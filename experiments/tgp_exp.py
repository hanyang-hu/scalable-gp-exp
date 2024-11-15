# Test the performance of TGP with Laplace / Normal priors

import torch, gpytorch
from scripts import models
from scripts.tgp import warping_func, CompositionalWarper
import linear_operator


class NeuralNetwork(torch.nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super(NeuralNetwork, self).__init__()
            layers = []
            current_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(torch.nn.Linear(current_dim, hidden_dim))
                layers.append(torch.nn.Tanh())
                current_dim = hidden_dim
            layers.append(torch.nn.Linear(current_dim, output_dim))
            self.network = torch.nn.Sequential(*layers)

            # Intialization
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.weight)
                    torch.nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.network(x)
    

if __name__ == '__main__':
    from uci_datasets import Dataset
    import argparse
    import tqdm

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', '-d', type=str, default='elevators', help="Name of the dataset, see https://github.com/treforevans/uci_datasets/blob/master/README.md for more details.")
    argparser.add_argument('--split', type=int, default=0, choices=range(0, 10), help="10-fold cross validation split of the dataset.")
    argparser.add_argument('--seed', type=int, default=42, help="Random seed, default to be 42.")
    argparser.add_argument('--prior', '-prior', type=str, default='Laplace', choices=['None', 'Laplace', 'Normal'], help="Subsampling method to use.")
    argparser.add_argument('--prior_weight', type=float, default=1e-5, help="Weights of the log prior term.")
    argparser.add_argument('--base_warper', type=str, default='sinh-arcsinh', choices=warping_func.keys(), help="Sequence of the warping functions.")
    argparser.add_argument('--warping_layers', type=int, default=20, help="Number of warping layers.")
    argparser.add_argument('--hidden_dims', nargs='+', default=None, help="Specification of the hidden layers.")
    argparser.add_argument('--exact_GP', type=bool, default=False, help="Use exact GP model.")
    argparser.add_argument('--keops', type=bool, default=False, help="Use KeOps for GP model.")
    argparser.add_argument('--iter', type=int, default=500, help="Number of iterations for training the GP model.")
    
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fetch the dataset
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

    # Standardize the data
    mean_X = X_train.mean(dim=0)
    std_X = X_train.std(dim=0) + 1e-7
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    mean_y = y_train.mean()
    std_y = y_train.std() + 1e-7
    y_train = (y_train - mean_y) / std_y
    y_test = (y_test - mean_y) / std_y

    # Specify the GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    if args.keops:
        model = models.ExactGPRBFKeOps(X_train, y_train, likelihood).to(device)
    else:
        model = models.ExactGPRBF(X_train, y_train, likelihood).to(device)

    likelihood.train()
    model.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with linear_operator.settings.max_cg_iterations(3000):
        if args.exact_GP:
            # Set up the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

            # Train the exact GP model
            progress_bar = tqdm.trange(args.iter)
            for i in progress_bar:
                optimizer.zero_grad()
                output = model(X_train) # model outputs
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({'Loss': loss.item()})

        else:
            # Build the warper and neural network
            warper_sequence = [args.base_warper] * args.warping_layers
            compose_warper = CompositionalWarper(warper_sequence)

            if args.hidden_dims is not None:
                args.hidden_dims = list(map(int, args.hidden_dims))
            else:
                args.hidden_dims = [1024]

            input_dim = X_train.shape[-1]
            output_dim = compose_warper.num_params

            mlp = NeuralNetwork(input_dim, args.hidden_dims, output_dim).to(device)

            # Set up the optimizer
            optimizer = torch.optim.Adam(
                [
                    {'params': model.parameters(), 'lr': 1e-1},
                    {'params': mlp.parameters(), 'lr': 1e-3},
                ]
            )

            # Train the TGP model
            progress_bar = tqdm.trange(args.iter)
            for i in progress_bar:
                optimizer.zero_grad()

                # Get model outputs
                output = model(X_train)
                target = compose_warper(mlp(X_train), y_train)

                # Compute the loss terms
                marginal_log_likelihood = -mll(output, target)
                log_warping_complexity = compose_warper.log_grad(mlp(X_train), y_train).mean()
                loss = marginal_log_likelihood - log_warping_complexity
                prior = torch.tensor(0.0).to(device)
                if args.prior == 'Laplace':
                    prior += args.prior_weight * mlp(X_train).norm(p=1, dim=-1).mean()
                if args.prior == 'Normal':
                    prior += args.prior_weight * mlp(X_train).norm(p=2, dim=-1).mean()
                loss += prior

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 10.0)
                
                loss.backward()

                optimizer.step()

                progress_bar.set_postfix(
                    {
                        'Loss': marginal_log_likelihood.item(), 
                        'Complexity': log_warping_complexity.item(),
                        'Prior': prior.item(),
                    }
                )

            model.set_train_data(X_train, compose_warper(mlp(X_train), y_train))

    # Test the model
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Obtain predictions on test set
        preds = likelihood(model(X_test))

        if args.exact_GP:
            rmse = torch.sqrt(torch.mean((preds.mean - y_test) ** 2)).item()
            nll = -torch.distributions.Normal(preds.mean, preds.stddev).log_prob(y_test).mean().item()
        else:
            inv_mean = compose_warper.inverse(mlp(X_test), preds.mean)
            rmse = torch.sqrt(torch.mean((inv_mean - y_test) ** 2)).item()

            warped_y_test = compose_warper(mlp(X_test), y_test)
            nll = -torch.distributions.Normal(preds.mean, preds.stddev).log_prob(warped_y_test).mean().item() - compose_warper.log_grad(mlp(X_test), y_test).mean().item()

    # Save the results
    # save the results to "./results/tgp_exp.json"
    # if the same experiment is run multiple times, replace the file with the new results
    import json
    import os

    if not os.path.exists('results'):
        os.makedirs('results')

    results = {
        'dataset': args.dataset,
        'split': args.split,
        'seed': args.seed,
        'prior': args.prior,
        'prior_weight': args.prior_weight,
        'base_warper': args.base_warper,
        'warping_layers': args.warping_layers,
        'hidden_dims': args.hidden_dims,
        'exact_GP': args.exact_GP,
        'iter': args.iter,
        'rmse': rmse,
        'nll': nll,
    }

    results_file = 'results/tgp_exp.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    # check if the same experiment has been run before
    same_exp = False
    for i in range(len(all_results)):
        if all([all_results[i][k] == results[k] for k in results.keys()]):
            all_results[i] = results
            same_exp = True
            break
    if not same_exp:
        all_results.append(results)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_file}")
    print(json.dumps(results, indent=4))