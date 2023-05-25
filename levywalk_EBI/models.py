import numpy as np

class Agent:
    def __init__(self, estimate_center, cov, β, delta=None):
        self.estimate_center = np.array([estimate_center]) 
        self.cov = cov
        self.β = β
        self.delta = delta
        self.delta_d = np.sqrt((2*np.pi)**2 * np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)
        self.orbit = estimate_center
        self.track = []

    def EMA(self, data):
        directions = self.β * (data - self.estimate_center)
        self.estimate_center += np.array(directions[0])
        movement_l = np.linalg.norm(directions)
        self.track.append(movement_l)
        self.orbit = np.vstack([self.orbit, self.estimate_center])

    def Min(self, data):
        data = np.array([data])
        A = data - self.estimate_center
        m_x, m_y = A[0]
        A = A.reshape(2, 1)
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.β * (1 - e[0,0]) / self.delta_d)
        directions = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        directions = directions.reshape(1, 2)
        X, Y = delta_P * directions[0]
        wight = m_x * Y / (m_y * X + m_x * Y)
        x_direction, y_direction = wight*delta_P*directions[0,0], (1-wight)*delta_P*directions[0,1]
        movement_l = np.linalg.norm(np.array([x_direction, y_direction]))
        if movement_l >= self.delta:
            x_direction, y_direction = 0.01 * (np.array([x_direction, y_direction]) / movement_l)
            movement_l = np.linalg.norm(np.array([x_direction, y_direction]))

            if  movement_l >= 0.01:
                movement_l = 0.01

        self.track.append(movement_l)
        self.estimate_center += np.array([x_direction, y_direction])
        self.orbit = np.vstack([self.orbit, self.estimate_center])    

    def Non_Min(self, data):
        theta = np.random.rand() * (np.pi*2)
        sin = np.sin(theta)
        cos = np.cos(theta)
        round_mat = np.array([[cos, -sin], [sin, cos]])
        round_inv = np.array([[cos, sin], [-sin, cos]])
        round_estimate_center = np.dot(round_mat, self.estimate_center.reshape(2, 1))
        round_data = np.dot(round_mat, np.array([data]).reshape(2, 1))
        A = round_data - round_estimate_center
        e = np.exp((-0.5) * np.dot(A.T, np.dot(self.inv_cov, A)))
        delta_P = (self.β * (1 - e[0,0]) / self.delta_d)
        directions = self.delta_d / (e[0,0] * np.dot(self.inv_cov, A))
        directions = directions.reshape(1, 2)
        wight = np.random.rand()
        round_x_direction, round_y_direction = wight*delta_P*directions[0,0], (1-wight)*delta_P*directions[0,1]
        x_direction, y_direction = np.dot(round_inv, np.array([[round_x_direction, round_y_direction]]).reshape(2, 1)).reshape(1, 2)[0]
        movement_l = np.linalg.norm(np.array([x_direction, y_direction]))
        if movement_l >= self.delta:
            x_direction, y_direction = 0.01 * (np.array([x_direction, y_direction]) / movement_l)
            movement_l = np.linalg.norm(np.array([x_direction, y_direction]))
            
            if movement_l >= 0.01:
                movement_l = 0.01

        self.track.append(movement_l)
        self.estimate_center += np.array([x_direction, y_direction])
        self.orbit = np.vstack([self.orbit, self.estimate_center])