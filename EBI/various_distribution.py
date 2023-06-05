import numpy as np

def Gaussian_distribution(generator_center, diagonal_matrix, step) -> list:
    sampling_data = np.random.multivariate_normal(generator_center, diagonal_matrix, step).T
    return sampling_data

def Uniform_distribution(generator_center, radius, step) -> list:
    theta = 2 * np.pi * np.random.rand(step)
    r = radius * np.sqrt(np.random.rand(step))
    sampling_data = np.array([generator_center[0] + r * np.cos(theta), generator_center[1] + r * np.sin(theta)])
    return sampling_data