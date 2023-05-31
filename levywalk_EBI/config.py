import numpy as np

samples = 300000  # Number of sample data
centers = np.array([0.3, 0.3])  # Center of generator distribution
covariance_matrix = np.array([[0.0025, 0], [0, 0.0025]])  # Gaussian covariance matrix
patch_radius = 0.05  # generation radius of uniform distribution

EMA_alpha = 0.00005  # EMA agent discount rate
BDI_alpha = 0.0001  # BDI agent discount rate
agent_center = np.array([0.0, 0.0])  # agent estimates
agent_covariance_matrix = np.array([[0.05, 0.0], [0.0, 0.05]])  # agent covariance matrix
delta = 0.01  # movement restrictions

s_priod = 200000
e_priod = 300000

