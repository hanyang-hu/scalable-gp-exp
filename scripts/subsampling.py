import torch
import math
from scripts import utils
import tqdm


"""
Random sampling to generate subsamples.
Inputs:
    X: torch.Tensor, shape (n_samples, n_features)
    n_samples: int, number of subsamples
"""
def random_sampling(X, n_samples):
    idx = torch.randperm(len(X))[:n_samples]
    return idx


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
    progress_bar = tqdm.tqdm(range(n_samples - 1), desc="Farthest Point Sampling")
    for _ in progress_bar:
        subsampled_X = X[subsamples].reshape(len(subsamples), d)
        dist = torch.min(
            dist, 
            torch.cdist(X, subsampled_X).min(dim=-1).values
        )
        l = torch.argmax(dist)
        subsamples.append(l.item())
    return subsamples


"""
Farthest point sampling (FPS) using KeOps to reduce the computational cost.
Inputs:
    X: torch.Tensor, shape (n_samples, n_features)
    n_samples: int, number of subsamples
"""
def farthest_point_sampling_keops(X, n_samples):
    _, d = X.shape

    # Define the expression to be computed in KeOps
    from pykeops.torch import Genred
    formula = "SqDist(x, y)"
    variables = [f"x = Vi({d})", f"y = Vj({d})"]
    routine = Genred(formula, variables, reduction_op="Min", axis=1)

    x_mean = X.mean(dim=0)
    l = torch.argmin((X - x_mean).norm(dim=-1))
    dist = routine(X, X[l].unsqueeze(0)).squeeze(-1)
    subsamples = [l.item()]
    progress_bar = tqdm.tqdm(range(n_samples - 1), desc="Farthest Point Sampling (KeOps)")
    for _ in progress_bar:
        subsampled_X = X[subsamples].reshape(len(subsamples), d)
        dist = torch.min(
            dist, 
            routine(X, subsampled_X).squeeze(-1)
        )
        l = torch.argmax(dist)
        subsamples.append(l.item())
    return subsamples


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
    tensor_grid = utils.get_adaptive_tensor_grid(int(n_samples * tau_factor), bounds).to(X.device)
    s = tensor_grid.shape[0]

    # initialze n_samples of G-sets
    G_sets = [torch.tensor([]).to(X.device) for _ in range(s)]
    for i in range(n):
        # find the closest point in the tensor grid
        dist = torch.cdist(X[i].unsqueeze(0), tensor_grid, p=float('inf')).squeeze()
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
        bounds_lst[i] = [(x_min[j] - 1e-7, x_max[j] + 1e-7) for j in range(d)]
        lebesgue_measure[i] = math.prod([b[1] - b[0] for b in bounds_lst[i]])

    # solve the apportionment problem based on Lebesgue measure
    set_size = utils.apportionment(n_samples, lebesgue_measure)
    
    for i in range(len(G_sets)):
        if set_size[i] >= 2:
            # generate low-discrepancy points in each G-set with size proportional to its Lebesgue measure
            low_discrepancy_sets[i] = utils.get_adaptive_tensor_grid(
                set_size[i], bounds_lst[i]
            ).to(X.device)
        elif set_size[i] == 1:
            low_discrepancy_sets[i] = G_sets[i].mean(dim=0).unsqueeze(0)
        else:
            low_discrepancy_sets[i] = torch.tensor([]).to(X.device)
        # print(low_discrepancy_sets[i].shape, set_size[i])

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
def anchor_net_method(X, n_samples, tau_factor=2):
    X = X.clone()
    anchor_net = get_anchor_net(X, n_samples, tau_factor)

    assert len(anchor_net) == n_samples

    subsampled_idx = []
    for i in range(n_samples):
        dist = torch.cdist(X, anchor_net[i].unsqueeze(0), p=float('inf')).min(dim=-1).values
        dist[subsampled_idx] = float('inf')
        idx = torch.argmin(dist)
        subsampled_idx.append(idx.item())
        
    return subsampled_idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a 2D dataset
    torch.manual_seed(0)
    # Generate a 2D dataset from some ellipsoids
    torch.manual_seed(0)
    num_points = 1000
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
    n_samples = 100
    fps_samples_idx = farthest_point_sampling(X, n_samples)
    fps_samples = X[fps_samples_idx]

    # Visualize the original dataset and FPS subsamples
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    random_samples_idx = random_sampling(X, n_samples)
    random_samples = X[random_samples_idx]
    plt.scatter(random_samples[:, 0], random_samples[:, 1], color='red', label='Random Subsamples')
    plt.title('Random Sampling')

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(fps_samples[:, 0], fps_samples[:, 1], color='red', label='FPS Subsamples')
    plt.title('Farthest Point Sampling')

    # Placeholder for anchor_net visualization
    tau_factor = 0.33

    anchor_samples_idx = anchor_net_method(X, n_samples, tau_factor=tau_factor)
    anchor_samples = X[anchor_samples_idx]

    # print("Anchor Net shape:", anchor_net.shape)
    # plt.subplot(1, 3, 2)
    # plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    # plt.scatter(anchor_net[:, 0], anchor_net[:, 1], color='red', label='Anchor Net', marker='x')
    # plt.title('Anchor Net')
    # plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(anchor_samples[:, 0], anchor_samples[:, 1], color='red', label='Anchor Net Subsamples')
    plt.title('Anchor Net Sampling')
    # plt.legend()

    plt.show()