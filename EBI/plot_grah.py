import matplotlib.pyplot as plt
import numpy as np
import config
plt.rcParams["legend.loc"] = "best"

def grah_show(agent, agent_type, sample_type, track, rank):
    fig_1 = plt.figure(figsize=(12, 6))
    fig_1.suptitle(f'Agent:{agent_type}\nDistribution:{sample_type}')
    ax1 = fig_1.add_subplot(1, 2, 1)
    ax2 = fig_1.add_subplot(1, 2, 2)

    fig_2 = plt.figure(figsize=(8, 8))
    fig_2.suptitle(f'Agent:{agent_type}\nDistribution:{sample_type}')
    ax3 = fig_2.add_subplot()

    ax1.plot(*agent.orbit[0:100000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
    ax1.scatter(*config.centers, color="orange", s=50, zorder=3, label="Center of data generating distribution")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.legend()
    ax1.set_title("step 0 ~ 100000")

    ax2.plot(*agent.orbit[config.s_priod:config.e_priod, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
    ax2.scatter(*agent.agent_center[0], color="red", s=50, zorder=4, label="The final mean estimate")
    ax2.scatter(*config.centers, color="orange", s=50, zorder=3, label="Center of data generating distribution")
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.legend()
    ax2.set_title(f"step {config.s_priod} ~ {config.e_priod}")

    x = np.linspace(1e-7, 1e-1, 100001)

    if agent_type == "Non_Min":

        if sample_type == "Gausian":
            y = 0.22*x**(-1)

        elif sample_type == "Uniform":
            y = 0.12*x**(-1)

        ax3.plot(x, y, label="y=a*x^(-1)")
        ax3.set_xscale('log')
        ax3.set_xlabel('Step length(log)')

    elif agent_type == "Min":

        if sample_type == "Gausian":
            y = 2e6*np.exp(-9e5*x)
            ax3.set_xlim([0,2e-5])
            ax3.plot(x, y, label="y=a*exp^(bx)")

        elif sample_type == "Uniform":
            ax3.set_xlim([0,3e-6])

        ax3.set_xlabel('Step length')
        ax3.set_ylim([1, 1e6])

    elif agent_type == "EMA":

        if sample_type == "Gausian":
            y = 1.5e7*np.exp(-1.27e6*x)
            ax3.plot(x, y, label="y=a*exp^(bx)")
            ax3.set_xlim([0,2e-5])

        elif sample_type == "Uniform":
            ax3.set_xlim([0,3e-6])

        ax3.set_xlabel('Step length')
        ax3.set_ylim([1, 1e6])

    ax3.plot(track, rank, label=f"{agent_type}")
    ax3.set_yscale('log')
    ax3.set_ylabel('Rank(log)')
    ax3.grid(which = "major", axis = "both", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
    ax3.legend()

    plt.show()