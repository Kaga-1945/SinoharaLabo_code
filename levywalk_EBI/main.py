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
        mu_x, mu_y = distance[0]
        self.mu += np.array([mu_x, mu_y])
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
        pass

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

step = 300000
#step = 200000
mu_true = np.array([0.3, 0.3])  # 生成元の平均パラメータ
cov_true = np.array([[0.0025, 0], [0, 0.0025]])  # 生成元の分散パラメータ
d = generate_data(mu_true, cov_true, step)  # データの生成
mu = np.array([0.0, 0.0])  # 平均の推定初期値
cov = np.array([[0.05, 0.0], [0.0, 0.05]]) 
#alpha_ebi = 0.0001
alpha_ema = 0.00005
delta = 0.01
#ebi = EBI(mu, cov, alpha_ebi, delta)
ema = EMA(mu, alpha_ema)
for i in range(step):
    ema.Ema(d[:, i])
    #ebi.Min(d[:, i])
    #ebi.Equal(d[:, i])
    #ebi.Rand(d[:, i])
"""
with open('result_Rand.bin', 'wb') as p:
    pickle.dump(EBI.track, p)
    pickle.dump(EBI.orbit, p)
"""
#track = np.array(ebi.track[200000:300000])
#track = np.array(ebi.track[100000:200000])
#track = np.array(ema.track[100000:200000])
track = np.array(ema.track[200000:300000])
#track = np.array(ebi.track)
rank = np.argsort(np.argsort(track))
track = np.sort(track)[::-1]
rank.sort()
fig_1 = plt.figure(figsize=(8, 8))
ax1 = fig_1.add_subplot()
#ax1.plot(*ebi.orbit[100000:200000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax1.plot(*ema.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
#ax1.plot(*ebi.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
#ax1.plot(*ebi.orbit.T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax1.scatter(*mu_true, color="orange", s=50, zorder=3, label="Center of date generating distribution")
#ax1.scatter(*ebi.mu[0], color="red", s=50, zorder=2, label="The final mean estimate")
ax1.scatter(*ema.mu[0], color="red", s=50, zorder=2, label="The final mean estimate")
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()
plt.show()

x = np.linspace(1e-7, 1e-1, len(rank))
y = 0.22*x**(-1)
fig_2 = plt.figure(figsize=(8, 8))
ax2 = fig_2.add_subplot()
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.plot(track, rank, label="Equal")
ax2.plot(x, y, label="y=0.22*x^(-1)")
ax2.legend()
plt.show()

"""
fig_3 = plt.figure(figsize=(8, 8))
ax3 = fig_3.add_subplot()
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(10000, 110000)
ax3.scatter(track[95000:100000], rank[95000:100000], label="0 ~ 5000")
ax3.legend()
plt.show()
"""