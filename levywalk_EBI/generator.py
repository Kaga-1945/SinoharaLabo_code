import numpy as np

def Gaussian_distribution(generator_center, diagonal_matrix, step) -> list:
    """
    2次元の正規分布を生成する
    center: 分布の中心座標
    diagonal_matrix: 共分散行列かつ対角行列
    step: 生成する乱数の個数
    """
    sampling_data = np.random.multivariate_normal(generator_center, diagonal_matrix, step).T
    return sampling_data

def Uniform_distribution(generator_center, radius, step) -> list:
    """
    2次元の正規分布を生成する
    center: 分布の中心座標
    radius: 分布の半径
    step: 生成する乱数の個数
    """
    theta = 2 * np.pi * np.random.rand(step)
    r = radius * np.sqrt(np.random.rand(step))
    sampling_data = np.array([generator_center[0] + r * np.cos(theta), generator_center[1] + r * np.sin(theta)])
    return sampling_data