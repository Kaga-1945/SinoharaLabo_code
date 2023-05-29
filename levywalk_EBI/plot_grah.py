import matplotlib.pyplot as plt
import numpy as np
import config
plt.rcParams["legend.loc"] = "best"

def grah_show(agent, agent_type, track, rank):
    fig_1 = plt.figure(figsize=(12, 6))
    ax1 = fig_1.add_subplot(1, 2, 1)
    ax2 = fig_1.add_subplot(1, 2, 2)

    fig_2 = plt.figure(figsize=(8, 8))
    ax3 = fig_2.add_subplot()

    ax1.plot(*agent.orbit[0:100000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
    ax1.scatter(*config.centers, color="orange", s=50, zorder=3, label="Center of data generating distribution")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.legend()
    ax1.set_title("step 0 ~ 100000")

    ax2.plot(*agent.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
    ax2.scatter(*agent.agent_center[0], color="red", s=50, zorder=4, label="The final mean estimate")
    ax2.scatter(*config.centers, color="orange", s=50, zorder=3, label="Center of data generating distribution")
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.legend()
    ax2.set_title("step 200000 ~ 300000")

    if agent_type == "Non_Min":
        x = np.linspace(1e-7, 1e-1, len(rank))
        y = 0.22*x**(-1)
        ax3.plot(x, y, label="y=0.22*x^(-1)")
    ax3.plot(track, rank, label=f"{agent_type}")
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()

    plt.show()