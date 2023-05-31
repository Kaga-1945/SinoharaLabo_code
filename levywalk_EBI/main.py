import numpy as np
import generate
import learning
import plot_grah
import config

if __name__ == "__main__":
    
    agent_types = ["EMA", "Min", "Non_Min"]
    sample_types = ["Gausian", "Uniform"]

    agent_type = agent_types[0]  # エージェントのタイプを指定
    sample_type = sample_types[0]  # 乱数のタイプを指定

    sampling_data = generate.sample(sample_type=sample_type)  # サンプルデータの生成

    agent = learning.inference(agent_type=agent_type, sampling_data=sampling_data)

    track = np.array(agent.track[config.s_priod:config.e_priod])
    track = np.sort(track)[::-1]  # 軌跡を降順にソートする
    rank = np.argsort(np.argsort(track))  # 上記を距離が長いもの順に順位づけする
    rank.sort()

    plot_grah.grah_show(agent=agent, agent_type=agent_type, sample_type=sample_type, track=track, rank=rank)