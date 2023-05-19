import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pickle

def generate_data(mu, cov, step):
    # 二次元の正規分布を生成
    d = np.random.multivariate_normal(mu, cov, step).T
    return d

class EMA:
    def __init__(self, mu, alpha):
        self.mu = np.array([mu])
        self.alpha = alpha
        self.orbit = mu
        self.track = []

    def Ema(self, d):
        distance = self.alpha * (d - self.mu)
        self.mu += np.array(distance[0])
        self.track.append(np.linalg.norm(distance))
        self.orbit = np.vstack([self.orbit, self.mu])
    

class EBI:
    def __init__(self, mu, cov, alpha, delta):
        self.mu = np.array([mu])  # 計算のために
        self.cov = cov
        self.alpha = alpha
        self.delta = delta
        self.orbit = mu
        self.delta_d = np.sqrt((2*np.pi)**2 * np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)
        self.track = []

    def Min(self, d):
        d = np.array([d])
        A = d - self.mu
        m_x, m_y = A[0]
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        X, Y = delta_P * direct[0]
        wight = m_x * Y / (m_y * X + m_x * Y)
        mu_x, mu_y = wight*delta_P*direct[0,0], (1-wight)*delta_P*direct[0,1]
        norm = np.linalg.norm(np.array([mu_x, mu_y]))
        if norm >= self.delta:
            mu_x, mu_y = 0.01 * (np.array([mu_x, mu_y]) / norm)
        distance = np.linalg.norm(np.array([mu_x, mu_y]))
        if distance >= 0.01:
            distance = 0.01
        self.track.append(distance)
        self.mu += np.array([mu_x, mu_y])
        self.orbit = np.vstack([self.orbit, self.mu])      

    def Equal(self, d):
        d = np.array([d])
        A = d - self.mu
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        mu_x, mu_y = (0.5*delta_P*direct)[0]
        norm = np.linalg.norm(np.array([mu_x, mu_y]))
        if norm >= self.delta:  # 修正量の補正
            mu_x, mu_y = 0.01 * (np.array([mu_x, mu_y]) / norm)
        distance = np.linalg.norm(np.array([mu_x, mu_y]))
        if distance >= 0.01:  # 誤差修正のため
            distance = 0.01
        self.track.append(distance)
        self.mu += np.array([mu_x, mu_y])
        self.orbit = np.vstack([self.orbit, self.mu])

    def Rand(self, d):
        d = np.array([d])
        A = d - self.mu
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        wight = np.random.rand()
        mu_x, mu_y = wight*delta_P*direct[0,0], (1-wight)*delta_P*direct[0,1]
        norm = np.linalg.norm(np.array([mu_x, mu_y]))
        if norm >= self.delta:
            mu_x, mu_y = 0.01 * (np.array([mu_x, mu_y]) / norm)
        distance = np.linalg.norm(np.array([mu_x, mu_y]))
        if distance >= 0.01:
            distance = 0.01
        self.track.append(distance)
        self.mu += np.array([mu_x, mu_y])
        self.orbit = np.vstack([self.orbit, self.mu])

    def Rand_2(self, d):
        theta = np.random.rand() * (np.pi * 2)
        sin = np.sin(theta)
        cos = np.cos(theta)
        round_mat = np.array([[cos, -sin], [sin, cos]])
        round_inv = np.array([[cos, sin], [-sin, cos]])
        round_mu = np.dot(round_mat, self.mu.reshape(2, 1))
        round_d = np.dot(round_mat, np.array([d]).reshape(2, 1))
        A = round_d - round_mu
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        wight = np.random.rand()
        mu_x, mu_y = wight*delta_P*direct[0,0], (1-wight)*delta_P*direct[0,1]
        mu_x, mu_y = np.dot(round_inv, np.array([[mu_x, mu_y]]).reshape(2, 1))
        mu_x = mu_x[0]
        mu_y = mu_y[0]
        norm = np.linalg.norm(np.array([mu_x, mu_y]))
        if norm >= self.delta:
            mu_x, mu_y = 0.01 * (np.array([mu_x, mu_y]) / norm)
        distance = np.linalg.norm(np.array([mu_x, mu_y]))
        if distance >= 0.01:
            distance = 0.01
        self.track.append(distance)
        self.mu += np.array([mu_x, mu_y])
        self.orbit = np.vstack([self.orbit, self.mu])

step = 300000
mu_true = np.array([0.3, 0.3])  # 生成元の平均パラメータ
cov_true = np.array([[0.0025, 0], [0, 0.0025]])  # 生成元の分散パラメータ
d = generate_data(mu_true, cov_true, step)  # データの生成
mu = np.array([0.0, 0.0])  # 平均の推定初期値
cov = np.array([[0.05, 0.0], [0.0, 0.05]]) 
alpha_ebi = 0.0001
alpha_ema = 0.00005
delta = 0.01
env = EBI(mu, cov, alpha_ebi, delta)
#env = EMA(mu, alpha_ema)
for i in range(step):
    #env.Ema(d[:, i])
    #env.Min(d[:, i])
    #env.Equal(d[:, i])
    #env.Rand(d[:, i])
    env.Rand_2(d[:, i])
"""
with open('result_Rand.bin', 'wb') as p:
    pickle.dump(EBI.track, p)
    pickle.dump(EBI.orbit, p)
"""
track = np.array(env.track[200000:300000])
rank = np.argsort(np.argsort(track))
track = np.sort(track)[::-1]
rank.sort()

fig_1 = plt.figure(figsize=(8, 8))
ax1 = fig_1.add_subplot()
ax1.plot(*env.orbit[0:100000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax1.scatter(*mu_true, color="orange", s=50, zorder=3, label="Center of data generating distribution")
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()
ax1.set_title("step 0 ~ 100000")
plt.show()

fig_2 = plt.figure(figsize=(8, 8))
ax2 = fig_2.add_subplot()
ax2.plot(*env.orbit[100000:200000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax2.scatter(*mu_true, color="orange", s=50, zorder=3, label="Center of data generating distribution")
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.legend()
ax2.set_title("step 100000 ~ 200000")
plt.show()

fig_3 = plt.figure(figsize=(8, 8))
ax3 = fig_3.add_subplot()
ax3.plot(*env.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax3.scatter(*env.mu[0], color="red", s=50, zorder=4, label="The final mean estimate")
ax3.scatter(*mu_true, color="orange", s=50, zorder=3, label="Center of data generating distribution")
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')
ax3.legend()
ax3.set_title("step 200000 ~ 300000")
plt.show()

x = np.linspace(1e-7, 1e-1, len(rank))
y = 0.22*x**(-1)
fig_4 = plt.figure(figsize=(8, 8))
ax4 = fig_4.add_subplot()
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.plot(track, rank, label="Equal")
ax4.plot(x, y, label="y=0.22*x^(-1)")
ax4.legend()
plt.show()