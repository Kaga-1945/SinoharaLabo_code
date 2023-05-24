import numpy as np

class Agent:
    def __init__(self, estimate_center, cov, alpha, delta=None):
        self.estimate_center = np.array([estimate_center]) 
        self.cov = cov
        self.alpha = alpha
        self.delta = delta
        self.delta_d = np.sqrt((2*np.pi)**2 * np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)
        self.orbit = estimate_center
        self.track = []

    def EMA(self, data):
        distance = self.alpha * (data - self.estimate_center)
        self.estimate_center += np.array(distance[0])
        self.track.append(np.linalg.norm(distance))
        self.orbit = np.vstack([self.orbit, self.estimate_center])

    def Min(self, d):
        d = np.array([d])
        A = d - self.estimate_center
        m_x, m_y = A[0]
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        X, Y = delta_P * direct[0]
        wight = m_x * Y / (m_y * X + m_x * Y)
        displacement_x, displacement_y = wight*delta_P*direct[0,0], (1-wight)*delta_P*direct[0,1]
        displacement_norm = np.linalg.norm(np.array([displacement_x, displacement_y]))
        if displacement_norm >= self.delta:
            displacement_x, displacement_y = 0.01 * (np.array([displacement_x, displacement_y]) / displacement_norm)

        distance = np.linalg.norm(np.array([displacement_x, displacement_y]))
        if distance >= 0.01:
            distance = 0.01

        self.track.append(distance)
        self.estimate_center += np.array([displacement_x, displacement_y])
        self.orbit = np.vstack([self.orbit, self.estimate_center])    

    def Non_Min(self, d):
        theta = np.random.rand() * (np.pi*2)
        sin = np.sin(theta)
        cos = np.cos(theta)
        round_mat = np.array([[cos, -sin], [sin, cos]])
        round_inv = np.array([[cos, sin], [-sin, cos]])
        round_estimate_center = np.dot(round_mat, self.estimate_center.reshape(2, 1))
        round_d = np.dot(round_mat, np.array([d]).reshape(2, 1))
        A = round_d - round_estimate_center
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.alpha * (1 - e[0,0]) / self.delta_d)
        direct = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        direct = direct.reshape(1, 2)
        wight = np.random.rand()
        round_displacement_x, round_displacement_y = wight*delta_P*direct[0,0], (1-wight)*delta_P*direct[0,1]
        displacement_x, displacement_y = np.dot(round_inv, np.array([[round_displacement_x, round_displacement_y]]).reshape(2, 1)).reshape(1, 2)[0]
        displacement_norm = np.linalg.norm(np.array([displacement_x, displacement_y]))
        if displacement_norm >= self.delta:
            displacement_x, displacement_y = 0.01 * (np.array([displacement_x, displacement_y]) / displacement_norm)

        distance = np.linalg.norm(np.array([displacement_x, displacement_y]))
        if distance >= 0.01:
            distance = 0.01

        self.track.append(distance)
        self.estimate_center += np.array([displacement_x, displacement_y])
        self.orbit = np.vstack([self.orbit, self.estimate_center])