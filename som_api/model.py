import numpy as np

def train_som(input_data, max_iterations: int, width: int, height: int):
    if input_data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (n_samples, n_features)")

    sigma_0 = max(width, height) / 2
    alpha_0 = 0.1
    weights = np.random.random((width, height, input_data.shape[1]))

    decay_constant = max_iterations / np.log(sigma_0)
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    for t in range(max_iterations):
        sigma_t = sigma_0 * np.exp(-t / decay_constant)
        alpha_t = alpha_0 * np.exp(-t / decay_constant)

        for v_t in input_data:
            diff = weights - v_t
            dist_sq = np.sum(diff ** 2, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dist_sq), (width, height))

            dx = xv - bmu_idx[0]
            dy = yv - bmu_idx[1]
            d2 = dx ** 2 + dy ** 2

            theta_t = np.exp(-d2 / (2 * sigma_t ** 2))
            influence = alpha_t * theta_t[..., np.newaxis]

            weights += influence * (v_t - weights)

    return weights
