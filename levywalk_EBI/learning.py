import numpy as np
import models
import config

def inference(agent_type, sampling_data):
    if agent_type == "EMA":
        agent = models.Agent(config.agent_center, config.agent_covariance, config.EMA_alpha)

        for i in range(config.samples):
            agent.EMA(sampling_data[:, i])

    elif agent_type == "Min" or agent_type == "Non_Min":
        agent = models.Agent(config.agent_center, config.agent_covariance, config.BDI_alpha, config.delta)

        if agent_type == "Min":
            for i in range(config.samples):
                agent.Min(sampling_data[:, i])

        elif agent_type == "Non_Min":
            for i in range(config.samples):
                agent.Non_Min(sampling_data[:, i])

    return agent