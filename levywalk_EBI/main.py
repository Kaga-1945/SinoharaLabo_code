import numpy as np
import generate
import learning
import plot_grah
import config

if __name__ == "__main__":
    
    agent_types = ["EMA", "Min", "Non_Min"]
    sample_types = ["Gausian", "Uniform"]

    agent_type = agent_types[2]  # Specify agent type
    sample_type = sample_types[0]  # Specify random number type

    sampling_data = generate.sample(sample_type=sample_type)  # Generate sample data

    agent = learning.inference(agent_type=agent_type, sampling_data=sampling_data)

    track = np.array(agent.track[config.s_priod:config.e_priod])
    track = np.sort(track)[::-1]  # Sort the trajectories in descending order
    rank = np.argsort(np.argsort(track))  # Arrange the above in descending order
    rank.sort()

    plot_grah.grah_show(agent=agent, agent_type=agent_type, sample_type=sample_type, track=track, rank=rank)