import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_data(mu, cov, step):
    # 二次元の正規分布を生成
    d = np.random.multivariate_normal(mu, cov, step).T
    return d

class EMA:
    def __init__(self, mu, cov, alpha, delta):
        self.mu = np.array([mu])  # 計算のために
        self.cov = cov
        self.alpha = alpha
        self.delta = delta
        self.orbit = mu
        self.delta_d = np.sqrt((2*np.pi)**2 * np.linalg.det(self.cov))
        self.track = []

    def Min(self, d):
        d = np.array([d])
        inv_cov = np.linalg.inv(self.cov)
        A = d - self.mu
        direct = A[0] / np.linalg.norm(np.array(A[0]))
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        mu_x, mu_y = delta_P*direct
        self.track.append(np.linalg.norm(np.array([mu_x, mu_y])))
        self.mu += np.array([mu_x, mu_y])
        self.orbit = np.vstack([self.orbit, self.mu])

    def Equal(self, d):
        d = np.array([d])
        inv_cov = np.linalg.inv(self.cov)
        A = d - self.mu
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(inv_cov, A))
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
        inv_cov = np.linalg.inv(self.cov)
        A = d - self.mu
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(inv_cov, A))
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
    
    def generate_data(self):
        # 二次元の正規分布を生成
        return np.random.multivariate_normal(self.mu[0], self.cov).T

step = 300000
mu1 = np.array([0.0, 0.0])  # 平均の推定初期値
cov1 = np.array([[0.0025, 0.0], [0.0, 0.0025]]) 
alpha1 = 0.0001
delta1 = 0.01
mu2 = np.array([1.0, 1.0])  # 平均の推定初期値
cov2 = np.array([[0.0025, 0.0], [0.0, 0.0025]]) 
alpha2 = 0.0001
delta2 = 0.01
ema_1 = EMA(mu1, cov1, alpha1, delta1)
ema_2 = EMA(mu2, cov2, alpha2, delta2)
for i in range(step):
    d1 = ema_1.generate_data()
    ema_2.Equal(d1)
    d2 = ema_2.generate_data()
    ema_1.Equal(d2)
"""
with open('result_Rand.bin', 'wb') as p:
    pickle.dump(ema.track, p)
    pickle.dump(ema.orbit, p)
"""
track_1 = np.array(ema_1.track[200000:300000])
#track_1 = np.array(ema_1.track)
rank_1 = np.argsort(np.argsort(track_1))
track_1 = np.sort(track_1)[::-1]
rank_1.sort()
track_2 = np.array(ema_2.track[200000:300000])
#track_2 = np.array(ema_2.track)
rank_2 = np.argsort(np.argsort(track_2))
track_2 = np.sort(track_2)[::-1]
rank_2.sort()
fig_1 = plt.figure(figsize=(8, 8))
ax1 = fig_1.add_subplot()
ax1.plot(*ema_1.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
#ax1.plot(*ema_1.orbit.T, color="blue", linewidth=1, zorder=1, label="Traject of the mean estimate")
ax1.scatter(*mu1, color="orange", s=50, zorder=3, label="Initialize agent's mean")
ax1.scatter(*ema_1.mu[0], color="red", s=50, zorder=2, label="The final mean estimate")
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()
plt.show()

x = np.linspace(1e-7, 1e-1, len(rank_1))
y = 11.0*x**(-0.85)
fig_2 = plt.figure(figsize=(8, 8))
ax2 = fig_2.add_subplot()
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.plot(track_1, rank_1, label="Agent1")
ax2.plot(track_2, rank_2, label="Agent2")
ax2.plot(x, y, label="y=11.0*x^(-0.85)")
ax2.legend()
plt.show()

fig_3 = plt.figure(figsize=(8, 8))
ax3 = fig_3.add_subplot()
ax3.plot(*ema_1.orbit[200000:300000, :].T, color="purple", linewidth=1, zorder=1, label="Traject of Agent1 mean estimate")
#ax3.plot(*ema_1.orbit.T, color="purple", linewidth=1, zorder=1, label="Traject of Agent1 mean estimate")
#ax3.scatter(*mu1, color="navy", s=50, zorder=5, label="Initialize Agent1 mean")
ax3.scatter(*ema_1.mu[0], color="red", s=50, zorder=3, label="The final Agent1 mean estimate")
ax3.plot(*ema_2.orbit[200000:300000, :].T, color="blue", linewidth=1, zorder=1, label="Traject of Agent2 mean estimate")
#ax3.plot(*ema_2.orbit.T, color="blue", linewidth=1, zorder=2, label="Traject of Agent2 mean estimate")
#ax3.scatter(*mu2, color="orange", s=50, zorder=6, label="Initialize Agent2 mean")
ax3.scatter(*ema_2.mu[0], color="red", s=50, zorder=4, label="The final Agent2 mean estimate")
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')
ax3.legend()
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