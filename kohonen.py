# kohonen.py

import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def train_som(input_data, max_iterations, width, height):
    
    """
    Trains a Self-Organizing Map (SOM) using Kohonen's algorithm.

    Args:
        input_data (np.ndarray): Input data with shape [n_samples, n_features].
        max_iterations (int): Number of training iterations.
        width (int): Width of the SOM grid.
        height (int): Height of the SOM grid.

    Returns:
        np.ndarray: The final trained SOM weight matrix (shape: [width, height, n_features]).
    """

    if input_data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (n_samples, n_features)")

    sigma_0 = max(width, height) / 2              # Initial neighborhood radius
    alpha_0 = 0.1                                 # Initial learning rate

    # Initialize weight vectors randomly in RGB space (3D)
    weights = np.random.random((width, height, input_data.shape[1]))

    # interaction constant for exponential decay
    decay_constant = max_iterations / np.log(sigma_0)

    # Generate meshgrid coordinates for the SOM grid
    # xv[i,j] = i, yv[i,j] = j for all i in [0, width), j in [0, height)
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    for t in range(max_iterations):
        sigma_t = sigma_0 * np.exp(-t / decay_constant)            # interaction-decayed neighborhood radius
        alpha_t = alpha_0 * np.exp(-t / decay_constant)            # interaction-decayed learning rate

        for v_t in input_data:
            # Find Best Matching Unit (BMU)
            diff = weights - v_t                    # Calculate the difference between weights and input vector
            dist_sq = np.sum(diff ** 2, axis=2)     # Squared Euclidean distance
            bmu_idx = np.unravel_index(np.argmin(dist_sq), (width, height)) # Find index of the BMU

            # Calculate squared distance from BMU to all units
            dx = xv - bmu_idx[0]
            dy = yv - bmu_idx[1]
            d2 = dx ** 2 + dy ** 2

            # Calculate the neighborhood function
            theta_t = np.exp(-d2 / (2 * sigma_t ** 2))

            # Compute influence (scalar) for each neuron, add dimension for broadcasting
            influence = alpha_t * theta_t[..., np.newaxis]

            # Update weights using vectorized operation
            weights += influence * (v_t - weights)

    return weights


def save_image(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, data)
    logging.info(f"Image saved to {output_path}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a Self-Organizing Map (Kohonen SOM)")
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--width', type=int, default=100, help='SOM grid width')
    parser.add_argument('--height', type=int, default=100, help='SOM grid height')
    parser.add_argument('--samples', type=int, default=10, help='Number of input samples')
    parser.add_argument('--features', type=int, default=3, help='Dimensionality of each input sample')
    parser.add_argument('--output', type=str, default='output/som.png', help='Path to save the output image')

    args = parser.parse_args()

    input_data = np.random.random((args.samples, args.features))
    logging.info(f"Training SOM with shape ({args.width}, {args.height}) for {args.iterations} iterations...")

    som_result = train_som(input_data, args.iterations, args.width, args.height)
    
    save_image(som_result, args.output)


if __name__ == '__main__':
    main()
