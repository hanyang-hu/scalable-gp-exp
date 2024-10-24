import torch
import math


"""
Generate low-discrepancy points using the adaptive tensor grid approach in a rectangular box.
To make sure the points are exactly n_samples, we add Sobol points to the grid in the end.
Inputs:
    n_samples: int, number of low-discrepancy points to generate
    bounds: list of tuples, bounds of the rectangular box
"""
def get_adaptive_tensor_grid(n_samples, bounds):
    n_dims = len(bounds)
    lengths = [b[1] - b[0] for b in bounds]
    lengths = [b[1] - b[0] for b in bounds]
    scaled_lengths = [length / math.prod(lengths) ** (1 / n_dims) for length in lengths]
    num_nodes = [max(math.floor(n_samples ** (1 / n_dims) * scaled_length), 1) for scaled_length in scaled_lengths]
    grid_lst = torch.meshgrid(
        *[torch.linspace(b[0] + (b[1] - b[0]) / (2 * num_nodes[i]), b[1] - (b[1] - b[0]) / (2 * num_nodes[i]), num_nodes[i]) for i, b in enumerate(bounds)],
        indexing="ij"
    )
    grid = torch.stack([g.flatten() for g in grid_lst], dim=-1)
    if n_samples > grid.shape[0]:
        sobol_points = torch.quasirandom.SobolEngine(n_dims, scramble=True).draw(int(n_samples - grid.shape[0]))
        sobol_points = sobol_points * torch.tensor([b[1] - b[0] for b in bounds])
        sobol_points = sobol_points + torch.tensor([b[0] for b in bounds])
        grid = torch.cat([grid, sobol_points], dim=0)
    return grid


"""
Solve the apportionment problem using the largest remainder.
Inputs:
    n_samples: int, total number of samples
    entitlement: list of floats, unnormalized proportions intended for each set
"""
def apportionment(n_samples, entitlement):
    total_entitlement = sum(entitlement)
    proportions = [e / total_entitlement for e in entitlement]
    set_size = [math.floor(n_samples * p) for p in proportions]
    remainder = [n_samples * p - s for p, s in zip(proportions, set_size)]
    _, remainder_idx = zip(*sorted(zip(remainder, range(len(remainder))), reverse=True))
    for i in range(n_samples - sum(set_size)):
        set_size[remainder_idx[i]] += 1
    return set_size