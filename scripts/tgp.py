import torch
import gpytorch


"""
The base Warper class.
Attributes:
    num_params: int, the number of parameters in the warping function.
    base_params: torch.Tensor, the default parameters of the warping function (set to ensure the function recovers identity).
    params: torch.Tensor, the batched parameters of the warping function (shape: [B, num_params]).
"""
class Warper():
    num_params = None
    base_params = None

    def __init__(self, params):
        self.params = params + self.base_params.to(params.device)

    def __call__(self, x):
        raise NotImplementedError
    
    def grad(self, x):
        raise NotImplementedError
    
    def inverse(self, y):
        raise NotImplementedError


class Affine(Warper):
    num_params = 2
    base_params = torch.tensor([1.0, 0.0]).requires_grad_(False)

    def __call__(self, x):
        return self.params[:,0] * x + self.params[:,1]
    
    def grad(self, x):
        return self.params[:,0]
    
    def inverse(self, y):
        return (y - self.params[:,1]) / self.params[:,0]
    

class Arcsinh(Warper):
    num_params = 4
    base_params = torch.tensor([0.0, 1.0, 0.0, 1.0]).requires_grad_(False)

    def __call__(self, x):
        return self.params[:,0] + self.params[:,1] * torch.asinh((x - self.params[:,2]) / self.params[:,3])
    
    def grad(self, x):
        return self.params[:,1] / torch.sqrt(self.params[:,3]**2 + (x - self.params[:,2])**2)
    
    def inverse(self, y):
        return self.params[:,2] + self.params[:,3] * torch.sinh((y - self.params[:,0]) / self.params[:,1])

class SinhArcsinh(Warper):
    num_params = 2
    base_params = torch.tensor([1.0, 0.0]).requires_grad_(False)

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
    
    def log_grad_mean(self, params, x):
        param_lst = torch.split(params, self.param_num_lst, dim=-1)
        # forward pass
        forward_cache = torch.zeros(len(self.warping_func_sequence), *x.shape).to(x.device)
        forward_cache[0] = x
        for i in range(1, len(self.warping_func_sequence)):
            warper = self.warping_func_sequence[i-1]
            forward_cache[i] += warper(param_lst[i-1])(x)
        # backward pass
        log_grad_sum = 0
        for i in range(len(self.warping_func_sequence)-1, -1, -1):
            warper = self.warping_func_sequence[i]
            log_grad_sum += torch.log(torch.clamp(warper(param_lst[i]).grad(forward_cache[i]), min=1e-7)).sum()
        return log_grad_sum / x.shape[0]
    
    def inverse(self, params, y):
        param_lst = torch.split(params, self.param_num_lst, dim=-1)
        for i in range(len(self.warping_func_sequence)-1, -1, -1):
            warper = self.warping_func_sequence[i]
            y = warper(param_lst[i]).inverse(y)
        return y