import torch
import math
from scripts import utils


"""
Farthest point sampling (FPS) to generate quasi-random subsamples.
Inputs:
    X: torch.Tensor, shape (n_samples, n_features)
    n_samples: int, number of subsamples
"""
def farthest_point_sampling(X, n_samples):
    _, d = X.shape
    x_mean = X.mean(dim=0)
    l = torch.argmin((X - x_mean).norm(dim=-1))
    dist = torch.cdist(X, X[l].unsqueeze(0)).squeeze()
    subsamples = [l.item()]
    for _ in range(n_samples - 1):
        subsampled_X = X[subsamples].reshape(len(subsamples), d)
        dist = torch.min(
            dist, 
            torch.cdist(X, subsampled_X).min(dim=-1).values
        )
        l = torch.argmax(dist)
        subsamples.append(l.item())
    return X[subsamples]


"""
Construct an anchor net for generating low-discrepancy points.
Inputs:
    X: torch.Tensor, shape (n_samples, n_features)
    n_samples: int, number of low-discrepancy points to generate
    tau_factor: int, number of G-sets is O(n_samples) controlled by tau_factor
"""
def get_anchor_net(X, n_samples, tau_factor):
    n, d = X.shape

    # find the smallest rectangular box that contains all the points
    x_min, _ = X.min(dim=0)
    x_max, _ = X.max(dim=0)

    # generate a low-discrepancy sequency of points in the box
    bounds = [(x_min[i], x_max[i]) for i in range(d)]
    tensor_grid = utils.get_adaptive_tensor_grid(n_samples * tau_factor, bounds)
    s = tensor_grid.shape[0]

    # initialze n_samples of G-sets
    G_sets = [torch.tensor([]) for _ in range(s)]
    for i in range(n):
        # find the closest point in the tensor grid
        dist = torch.cdist(X[i].unsqueeze(0), tensor_grid).squeeze()
        idx = torch.argmin(dist)
        # add the point to the corresponding G-set
        G_sets[idx] = torch.cat([G_sets[idx], X[i].unsqueeze(0)], dim=0)
    # keep only the non-empty G-sets
    G_sets = [G_set for G_set in G_sets if len(G_set) > 0]

    low_discrepancy_sets = [None] * len(G_sets)
    bounds_lst = [None] * len(G_sets)
    lebesgue_measure = [None] * len(G_sets)

    # find the smallest rectangular box that contains all the points in each G-set
    # compute their corresponding Lebesgue measure
    for i in range(len(G_sets)):
        x_min, _ = G_sets[i].min(dim=0)
        x_max, _ = G_sets[i].max(dim=0)
        bounds_lst[i] = [(x_min[j] - 1e-6, x_max[j] + 1e-6) for j in range(d)]
        lebesgue_measure[i] = math.prod([b[1] - b[0] for b in bounds_lst[i]])

    # solve the apportionment problem based on Lebesgue measure
    set_size = utils.apportionment(n_samples, lebesgue_measure)
    
    for i in range(len(G_sets)):
        # generate low-discrepancy points in each G-set with size proportional to its Lebesgue measure
        if lebesgue_measure[i] != 0:
            low_discrepancy_sets[i] = utils.get_adaptive_tensor_grid(
                set_size[i], bounds_lst[i]
            )
        else:
            low_discrepancy_sets[i] = torch.tensor([])

    # concatenate all the low-discrepancy points
    low_discrepancy_points = torch.cat(low_discrepancy_sets, dim=0)
    return low_discrepancy_points


"""
Generate low-discrepancy points using the anchor net approach.
Inputs:
    X: torch.Tensor, shape (n_samples, n_features)
    n_samples: int, number of low-discrepancy points to generate
    tau_factor: int, number of G-sets is O(n_samples) controlled by tau_factor
"""
def anchor_net_method(X, n_samples, tau_factor=1):
    anchor_net = get_anchor_net(X, n_samples, tau_factor)
    subsampled_X = []
    for i in range(n_samples):
        dist = torch.cdist(X, anchor_net[i].unsqueeze(0), p=float('inf')).min(dim=-1).values
        idx = torch.argmin(dist)
        subsampled_X.append(X[idx])
    return torch.stack(subsampled_X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a 2D dataset
    torch.manual_seed(0)
    # Generate a 2D dataset from some ellipsoids
    torch.manual_seed(0)
    num_points = 100
    num_ellipsoids = 3
    X = []

    for _ in range(num_ellipsoids):
        center = torch.rand(2) * 10  # Random center for each ellipsoid
        axes = torch.rand(2) * 2 + 1  # Random axes lengths
        angle = torch.rand(1).item() * 2 * torch.pi  # Random rotation angle
        angle = torch.tensor(angle)

        # Rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ])

        for _ in range(num_points // num_ellipsoids):
            point = torch.rand(2) * 2 - 1  # Random point in [-1, 1] x [-1, 1]
            point = point * axes  # Scale by axes lengths
            point = rotation_matrix @ point  # Rotate point
            point = point + center  # Translate to center
            X.append(point)

    X = torch.stack(X)

    # Apply farthest point sampling
    n_samples = 50
    fps_samples = farthest_point_sampling(X, n_samples)

    # Visualize the original dataset and FPS subsamples
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(fps_samples[:, 0], fps_samples[:, 1], color='red', label='FPS Subsamples')
    plt.title('Farthest Point Sampling')
    plt.legend()

    # Placeholder for anchor_net visualization
    n_samples = 50
    anchor_samples = anchor_net_method(X, n_samples, tau_factor=0.5)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(anchor_samples[:, 0], anchor_samples[:, 1], color='red', label='Anchor Net Subsamples')
    plt.title('Anchor Net Sampling')
    plt.legend()

    plt.show()