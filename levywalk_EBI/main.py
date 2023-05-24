import numpy as np
import generator
import models  
import plot_grah

if __name__ == "__main__":

    agent_types = ["EMA", "Min", "Non_Min"]
    sample_types = ["Gausian", "Uniform"]

    agent_type = agent_types[2]  # エージェントのタイプを指定
    sample_type = sample_types[0]  # 乱数のタイプを指定

    step = 300000  # 生成するデータ数
    generator_centor = np.array([0.3, 0.3])  # 生成元の中心

    if sample_type == "Gausian":
        generator_covariance = np.array([[0.0025, 0], [0, 0.0025]])  # 生成元の分散パラメータ
        sampling_data = generator.Gaussian_distribution(generator_centor, generator_covariance, step)  # データの生成

    elif sample_type == "Uniform":
        generator_radius = 0.05  # 生成半径
        sampling_data = generator.Uniform_distribution(generator_centor, generator_radius, step)  # データの生成

    estimate_center = np.array([0.0, 0.0])  # エージェントの中心の推定値の初期値
    agent_covariance = np.array([[0.05, 0.0], [0.0, 0.05]])  # エージェントの共分散の推定値の初期値(今回は固定)

    if agent_type == "EMA":
        alpha = 0.00005  # 割引率
        agent = models.Agent(estimate_center, agent_covariance, alpha)

        for i in range(step):
            agent.EMA(sampling_data[:, i])

    elif agent_type == "Min" or agent_type == "Non_Min":
        alpha = 0.0001  # 割引率
        delta = 0.01  # 修正上限値
        agent = models.Agent(estimate_center, agent_covariance, alpha, delta)

        if agent_type == "Min":
            for i in range(step):
                agent.Min(sampling_data[:, i])

        elif agent_type == "Non_Min":
            for i in range(step):
                agent.Non_Min(sampling_data[:, i])

    track = np.array(agent.track[200000:300000])  # 200000 ~ 300000ステップの推移の軌跡
    track = np.sort(track)[::-1]  # 軌跡を降順にソートする
    rank = np.argsort(np.argsort(track))  # 上記を距離が長いもの順に順位づけする
    rank.sort()

    plot_grah.grah_show(agent, agent_type, generator_centor, track, rank)