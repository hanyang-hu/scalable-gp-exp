import torch
import gpytorch


"""
The base Warper class.
Attributes:
    num_params: int, the number of parameters in the warping function.
    params: torch.Tensor, the batched parameters of the warping function (shape: [B, num_params]).
"""
class Warper():
    num_params = None

    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        raise NotImplementedError
    
    def grad(self, x):
        raise NotImplementedError
    
    def inverse(self, y):
        raise NotImplementedError


class Affine(Warper):
    num_params = 2

    def __call__(self, x):
        return self.params[:,0] * x + self.params[:,1]
    
    def grad(self, x):
        return self.params[:,0]
    
    def inverse(self, y):
        return (y - self.params[:,1]) / self.params[:,0]
    

class Arcsinh(Warper):
    num_params = 4

    def __call__(self, x):
        return self.params[:,0] + self.params[:,1] * torch.asinh((x - self.params[:,2]) / self.params[:,3])
    
    def grad(self, x):
        return self.params[:,1] / torch.sqrt(self.params[:,3]**2 + (x - self.params[:,2])**2)
    
    def inverse(self, y):
        return self.params[:,2] + self.params[:,3] * torch.sinh((y - self.params[:,0]) / self.params[:,1])

class SinhArcsinh(Warper):
    num_params = 2

    def __call__(self, x):
        return torch.sinh(self.params[:,0] * torch.asinh(x) - self.params[:,1])
    
    def grad(self, x):
        return self.params[:,0] * torch.cosh(self.params[:,0] * torch.asinh(x) - self.params[:,1]) / torch.sqrt(1 + x**2)
    
    def inverse(self, y):
        return torch.sinh((torch.asinh(y) + self.params[:,1]) / self.params[:,0])


warping_func = {
    "affine" : Affine,
    # "symlog" : SymLog,
    "arcsinh" : Arcsinh,
    # "box_cox" : BoxCox,
    "sinh-arcsinh" : SinhArcsinh
}


"""
Compose different base warpers.
Use the chain rule (backpropagation) to compute the gradient.
"""
class CompositionalWarper():
    def __init__(self, warping_sequence):

        self.warping_func_sequence = [warping_func[warper_name] for warper_name in warping_sequence]
        self.param_num_lst = []
        for warper in self.warping_func_sequence:
            self.param_num_lst.append(warper.num_params)
        self.num_params = sum(self.param_num_lst)

    def __call__(self, params, x):
        param_lst = torch.split(params, self.param_num_lst, dim=-1)
        for i in range(len(self.warping_func_sequence)):
            warper = self.warping_func_sequence[i]
            x = warper(param_lst[i])(x)
        return x
    
    def grad(self, params, x):
        param_lst = torch.split(params, self.param_num_lst, dim=-1)
        # forward pass
        forward_cache = torch.zeros(len(self.warping_func_sequence), *x.shape).to(x.device)
        forward_cache[0] = x
        for i in range(1, len(self.warping_func_sequence)):
            warper = self.warping_func_sequence[i-1]
            forward_cache[i] += warper(param_lst[i-1])(x)
        # backward pass
        grad = torch.ones_like(x)
        for i in range(len(self.warping_func_sequence)-1, -1, -1):
            warper = self.warping_func_sequence[i]
            grad = grad * warper(param_lst[i]).grad(forward_cache[i])
        return grad
    
    def inverse(self, params, y):
        param_lst = torch.split(params, self.param_num_lst, dim=-1)
        for i in range(len(self.warping_func_sequence)-1, -1, -1):
            warper = self.warping_func_sequence[i]
            y = warper(param_lst[i]).inverse(y)
        return y


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=5/2, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm

    # Generate synthetic data that is hard to be modeled by a standard GP
    np.random.seed(0)
    true_func = lambda x : np.sin(x) + 0.3 * np.sin(3*x)
    x = np.random.uniform(-10, 10, 2000)
    x = x[(x < -7) | ((x > -3) & (x < 4)) | (8 < x)]
    y = true_func(x) + 0.05 * np.random.normal(0, 1, len(x)) * abs(x)

    # Convert to torch tensors
    train_x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(y, dtype=torch.float32)

    # Define the compositional warper module
    m=20
    warping_sequence = ["sinh-arcsinh"] * m
    compose_warper = CompositionalWarper(warping_sequence)

    params = torch.tensor([1.0, 0.0], dtype=torch.float32).repeat(m).unsqueeze(0)
    params = params.to(train_x.device).requires_grad_(True)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, params.shape[-1])
    )

    # Initialize weights of the MLP
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    mlp.apply(init_weights)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': 1e-2},
            {'params': [params], 'lr': 1e-5},
            {'params': mlp.parameters(), 'lr': 1e-5}
        ],
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Print 

    training_iter = 3000
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    alpha = 1 # weight for the warping complexity
    progress_bar = tqdm.tqdm(range(training_iter), desc="Training")
    for i in progress_bar:
        optimizer.zero_grad()
        output = model(train_x)
        target = compose_warper(mlp(train_x)+params, train_y)
        model.set_train_data(train_x, target)
        gaussian_loss = -mll(output, target) 
        # param_grad = torch.autograd.grad(gaussian_loss, params, retain_graph=True)[0]
        warping_complexity = torch.log(torch.clamp(compose_warper.grad(mlp(train_x)+params, train_y), min=1e-7)).sum()
        loss = gaussian_loss - alpha * warping_complexity
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0, error_if_nonfinite=True)
        # torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'Gaussian Loss': gaussian_loss.item(), 'Warping Complexity': warping_complexity.item()})

    model2 = ExactGPModel(train_x, train_y, likelihood)

    model2.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model2)
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-2)

    training_iter = 200
    progress_bar = tqdm.tqdm(range(training_iter), desc="Training")
    for i in progress_bar:
        optimizer.zero_grad()
        output = model2(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'Loss': loss.item()})

    model2.eval()


    # # Plot the compositional warping function
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # x_left = np.min(y)
    # x_right = np.max(y)
    # test_x = torch.linspace(x_left, x_right, 1000).unsqueeze(-1)
    # test_y = compose_warper(params, test_x).detach()
    # plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy())
    # plt.title("Compositional Warping Function")

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # print("Lengthscale: ", model.covar_module.base_kernel.lengthscale)

    # Plot the predictive distribution
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(-10, 10, 1000).unsqueeze(-1)
        observed_pred = likelihood(model(test_x))
        post_mean = observed_pred.mean
        post_var = observed_pred.variance
        upper, lower = post_mean + 2 * post_var.sqrt(), post_mean - 2 * post_var.sqrt()
        post_mean = compose_warper.inverse(mlp(test_x)+params, post_mean)
        upper = compose_warper.inverse(mlp(test_x)+params, upper)
        lower = compose_warper.inverse(mlp(test_x)+params, lower)
        # plt.subplot(1, 2, 2)
        plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k.')
        plt.plot(test_x.cpu().numpy(), true_func(test_x.cpu().numpy()), 'black', linestyle='--')
        test_x = test_x.squeeze(-1)
        
        post_mean2 = model2(test_x).mean
        post_var2 = model2(test_x).variance
        upper2, lower2 = post_mean2 + 2 * post_var2.sqrt(), post_mean2 - 2 * post_var2.sqrt()
        plt.plot(test_x.cpu().numpy(), post_mean2.cpu().numpy(), 'r', label='Prediction of Exact GP')
        plt.fill_between(test_x.cpu().numpy(), upper2.cpu().numpy(), lower2.cpu().numpy(), alpha=0.5, color='r')

        # print(test_x.shape, post_mean.shape, upper.shape, lower.shape)
        plt.plot(test_x.cpu().numpy(), post_mean.cpu().numpy(), 'b', label='Prediction of TGP')
        plt.fill_between(test_x.cpu().numpy(), upper.cpu().numpy(), lower.cpu().numpy(), alpha=0.5, color='b')

        plt.legend()

        plt.title("Predictive Distribution")
        plt.show()

    # # Sample function from the posterior
    # with torch.no_grad(), gpytorch.settings.ciq_samples(False):
    #     test_x = torch.linspace(-5, 5, 2000).unsqueeze(-1)
    #     observed_pred = likelihood(model(test_x))
    #     post_func_lst = []
    #     for i in range(5):
    #         post_func = observed_pred.rsample()
    #         post_func = compose_warper.inverse(params, post_func)
    #         post_func_lst.append(post_func)
    
    # # Plot the posterior samples
    # plt.ylim(-1.0, 1.0)
    # plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k.')
    # for i in range(len(post_func_lst)):
    #     plt.plot(test_x.numpy(), post_func_lst[i].numpy(), label=f'Posterior Sample {i}')
    # plt.show()
        
