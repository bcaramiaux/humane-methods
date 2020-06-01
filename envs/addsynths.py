import numpy as np



class AdditiveSynthesisEnv():
    # Gola: reach this sound target: 
    # [0.964286, 0.892857, 0.755814, 0.674419, 0.053571, 0.6875, 0.669643, 0.017857, 0.508929, 0.348837, 0.544643, 0.4375, 0.571429, 0.209302, 0.186047, 0.5625, 0.285714, 0.116279, 0.571429, 0.428571]
    def __init__(self):
        self.goal = [0.965116, 0.755814, 0.581395, 0.488372, 0.383721, 0.302326, 0.244186, 0.186047, 0.151163, 0.081395]
        self.n_dim = len(self.goal)
        self.n_quant = 10
        self.n_states = self.n_dim   # 10 ^ 20 (!!)
        self.n_actions = 2 * self.n_dim # +1/-1 for i in [1,20] => (0, 1, .., 9) = + 1 (i=1..10) ; (10, 11, .., 19) = - 1 (i=1..10) 
        self.state = np.random.rand(self.n_dim)

    def actions(self, s):
        #coords = self.state_to_coord[s]
        #acts = np.where(self.transition[coords[0], coords[1], :] != 0)[0]
        valid_acts = []
        for i in range(self.n_dim):
            if s[0,i] + 1/self.n_quant < 1.0:
                valid_acts.append(i)
        for i in range(self.n_dim):
            if s[0,i] - 1/self.n_quant > 0.0:
                valid_acts.append(self.n_dim + i)
        #print(s, valid_acts)
        return valid_acts

    def step(self, action):
        s = self.state
        dist = np.sqrt(np.sum([(s[0,i] - self.goal[i])**2 for i in range(self.n_dim)]))/self.n_dim
        if len(np.where(s[0,:] > 1.0)[0]): 
            reward = 0.0
            for i in np.where(s[0,:] > 1.0)[0]:
                reward += (1.0 - s[0,i]) * 1000.0
        elif len(np.where(s[0,:] < 0.0)[0]):
            reward = 0.0
            for i in np.where(s[0,:] < 0.0)[0]:
                reward += s[0,i] * 1000.0
        else:
            reward = abs(((1.0 - dist) ** 2)) * 100.0
        if action < self.n_dim:
            next_s = np.copy(s)
            next_s[0, action] += 1/self.n_quant
        elif action >= self.n_dim:
            next_s = np.copy(s)
            next_s[0, action - self.n_dim] -= 1/self.n_quant
        done = self._is_final(next_s)
        return (next_s, reward, done)
    
    def reset(self):
        self.state = np.random.rand(self.n_dim) #self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state)

    def render(self, mode='human'):
        return 

    def close(self):
        return 

    def _is_final(self, s):
        dist = np.sqrt(np.sum([(s[0,i] - self.goal[i])**2 for i in range(self.n_dim)]))/self.n_dim
        if dist < 0.01:
            return True
        else:
            return False