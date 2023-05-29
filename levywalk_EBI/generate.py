import numpy as np
import config
import various_distribution

def sample(sample_type):
    if sample_type == "Gausian":
        sampling_data = various_distribution.Gaussian_distribution(config.centers, config.covariance_matrix, config.samples)  # データの生成

    elif sample_type == "Uniform":
        sampling_data = various_distribution.Uniform_distribution(config.centers, config.patch_radius, config.samples)  # データの生成

    return sampling_data