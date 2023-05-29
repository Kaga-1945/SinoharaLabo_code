import numpy as np

samples = 300000  # 生成するサンプルデータの数
centers = np.array([0.3, 0.3])  # 生成分布の中心
covariance_matrix = np.array([[0.0025, 0], [0, 0.0025]])  # ガウス分布の共分散行列
patch_radius = 0.05  # 一様分布の生成半径

EMA_alpha = 0.00005  # EMAエージェントの割引率
BDI_alpha = 0.0001  # BDIエージェントの割引率
agent_center = np.array([0.0, 0.0])  # エージェントの推定値
agent_covariance = np.array([[0.05, 0.0], [0.0, 0.05]])  # エージェントが持つ共分散行列
delta = 0.01  # 移動量の上限値

