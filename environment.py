#!/usr/bin/env python

"""
Copyright 2019 Baptiste Caramiaux 
"""



import numpy as np

class ToyExample:
    # Problem: start at 0, go to G, avoid Xs
    # ---------------------------------------
    # '0 _ _ _ _ _ _'
    # '_ X _ _ _ _ _'
    # '_ _ _ X X _ _'
    # '_ _ X _ G _ _'
    # '_ _ _ _ _ _ _'
    #
    # ' 0  1  2  3  4  5  6'
    # ' 7  8  9 10 11 12 13'
    # '14 15 16 17 18 19 20'
    # '21 22 23 24 25 26 27'
    # '28 29 30 31 32 33 34'
    #
    def __init__(self):
        self.grid_r = 5
        self.grid_c = 7
        self.n_states = self.grid_r * self.grid_c
        self.n_actions = 4 # up, right, down, left
        trans = np.zeros((self.grid_r, self.grid_c, 4))
        for i in range(trans.shape[1]):
            if i == 0:
                trans[:, i, :] = 1
                trans[:, i, 3] = 0
                trans[0, i, 0] = 0
                trans[-1, i, 2] = 0
            elif i == trans.shape[1]-1:
                trans[:, i, :] = 1
                trans[:, i, 1] = 0
                trans[0, i, 0] = 0
                trans[-1, i, 2] = 0
            else:
                trans[:, i, :] = 1
                trans[0, i, 0] = 0
                trans[-1, i, 2] = 0
        # X_coord = [[1,1], [3,2], [2,3], [2,4]]
        # for X in X_coord:
        #     X_x = X[0]
        #     X_y = X[1]
        #     trans[X_x-1, X_y, 2] = 0
        #     trans[X_x, X_y-1, 1] = 0
        #     trans[X_x+1, X_y, 0] = 0
        #     trans[X_x, X_y+1, 3] = 0
        self.transition = trans

        self.state_to_coord = []
        for i in range(self.grid_r):
            for j in range(self.grid_c):
                self.state_to_coord.append([i,j])
        
        self.rewards = np.copy(trans)
        self.rewards[3,3,1] = 50
        self.rewards[3,5,3] = 50
        self.rewards[4,4,0] = 50
        for i1 in range(self.rewards.shape[0]):
            for i2 in range(self.rewards.shape[1]):
                for i3 in range(self.rewards.shape[2]):
                    if self.rewards[i1, i2, i3] == 0:
                        self.rewards[i1, i2, i3] = -1000
        X_coord = [[1,1], [3,2], [2,3], [2,4]]
        for X in X_coord:
            X_x = X[0]
            X_y = X[1]
            self.rewards[X_x-1, X_y, 2] = -100
            self.rewards[X_x, X_y-1, 1] = -100
            self.rewards[X_x+1, X_y, 0] = -100
            self.rewards[X_x, X_y+1, 3] = -100


    def sample_initial_state(self):
        s = 0 #np.random.randint(low=0, high=self.n_states)
        return s

    def actions(self, s):
        coords = self.state_to_coord[s]
        acts = np.where(self.transition[coords[0], coords[1], :] != 0)[0]
        # acts = [0, 1, 2, 3]
        # print(s, acts)
        return acts

    def state_reward(self, s, a):
        coords = self.state_to_coord[s]
        if a == 0:
            next_s = s - self.grid_c
        elif a == 1:
            next_s = s + 1
        elif a == 2:
            next_s = s + self.grid_c
        elif a == 3:
            next_s = s - 1
        done = self.is_final(next_s)
        return (next_s, self.rewards[coords[0], coords[1], a], done)

    def is_final(self, s):
        coords = self.state_to_coord[s]
        if coords[0] == 3 and coords[1] == 4:
            return True
        else:
            return False


class HumaneMethodsEnvironmentDiscrete:
    # Problem: start at 0, go to G, avoid Xs
    def __init__(self):

        self.scene_names = ["stool",
                            "ribs",
                            "cross",
                            "naked cage",
                            "marionnette",
                            "frame",
                            "costume desoriented",
                            "costume branch",
                            "costume amygdala shake",
                            "turn on amygdala"]
        # self.state_names = ["stool",
        #                     "frame",
        #                     "turn on amygdala"]

        self.bodies = 2
        self.action_names = ["still",
                             "low energy, moving",
                             "high energy, fast rate",
                             "high energy, low rate"]
        

        self.n_actions = len(self.scene_names)
        self.n_states = self.bodies ** len(self.action_names)

        # self.n_scenes = len(self.scene_names)
        # self.n_states = self.n_scenes
        #                     # 0: stool
        #                     # 1: ribs
        #                     # 2: cross
        #                     # 3: naked cage
        #                     # 4: marionnette
        #                     # 5: frame
        #                     # 6: costume desoriented
        #                     # 7: costume branch
        #                     # 8: costume amygdala shake
        
        
        # self.n_actions = 4  # 0: still
        #                     # 1: low energy, moving
        #                     # 2: high energy, fast rate
        #                     # 3: high energy, low rate 

        

        # trans_shape = self.bodies * (self.body_states,) + (self.n_actions,)

        levels = 4
        self.state_to_coord = []
        for i1 in range(levels):
            for i2 in range(levels):
                self.state_to_coord.append([i1, i2])


        self.transitions = np.ones((self.n_states, self.n_actions))
        # self.transitions[0, 1] = 9   # get to 9
        # self.transitions[0, 3] = 5   # get to 5

        # self.transitions = np.zeros((self.n_states, self.n_actions))
        
        # for i in range(trans.shape[1]):
        #     if i == 0:
        #         trans[:, i, :] = 1
        #         trans[:, i, 3] = 0
        #         trans[0, i, 0] = 0
        #         trans[-1, i, 2] = 0
        #     elif i == trans.shape[1]-1:
        #         trans[:, i, :] = 1
        #         trans[:, i, 1] = 0
        #         trans[0, i, 0] = 0
        #         trans[-1, i, 2] = 0
        #     else:
        #         trans[:, i, :] = 1
        #         trans[0, i, 0] = 0
        #         trans[-1, i, 2] = 0
        # X_coord = [[1,1], [3,2], [2,3], [2,4]]
        # for X in X_coord:
        #     X_x = X[0]
        #     X_y = X[1]
        #     trans[X_x-1, X_y, 2] = 0
        #     trans[X_x, X_y-1, 1] = 0
        #     trans[X_x+1, X_y, 0] = 0
        #     trans[X_x, X_y+1, 3] = 0
        # self.transition = trans

        

        # for i in range(self.grid_r):
        #     for j in range(self.grid_c):
        #         self.state_to_coord.append([i,j])
        
        # self.rewards = np.copy(trans)
        # self.rewards[3,3,1] = 50
        # self.rewards[3,5,3] = 50
        # self.rewards[4,4,0] = 50
        # for i1 in range(self.rewards.shape[0]):
        #     for i2 in range(self.rewards.shape[1]):
        #         for i3 in range(self.rewards.shape[2]):
        #             if self.rewards[i1, i2, i3] == 0:
        #                 self.rewards[i1, i2, i3] = -1000
        # X_coord = [[1,1], [3,2], [2,3], [2,4]]
        # for X in X_coord:
        #     X_x = X[0]
        #     X_y = X[1]
        #     self.rewards[X_x-1, X_y, 2] = -100
        #     self.rewards[X_x, X_y-1, 1] = -100
        #     self.rewards[X_x+1, X_y, 0] = -100
        #     self.rewards[X_x, X_y+1, 3] = -100

    def get_state_name(self, id):
        return self.scene_names[id]

    def coord_to_state(self, vector):
        idx = -1
        for i in range(len(self.state_to_coord)):
            equals = True
            for j in range(len(self.state_to_coord[i])):
                if vector[j] != self.state_to_coord[i][j]:
                    equals = False
            if equals:
                idx = i
                break
        return idx


    def sample_initial_state(self):
        #s = 0 #np.random.randint(low=0, high=self.n_states)
        # s = list(self.bodies * (0,))
        # s = [0 for i in range(self.bodies)]
        s = 0   # [0, 0, 0, 0]
        return s

    def actions(self, s):
        # coords = self.state_to_coord[s]
        # acts = np.where(self.transition[coords[0], coords[1], :] != 0)[0]

        # acts = np.arange(self.n_actions)
        acts = np.where(self.transitions[s,:] != 0)[0]

        # acts = [0, 1, 2, 3]
        # print(s, acts)
        return acts

    def state_reward(self, s, a, next_s):
        # coords = self.state_to_coord[s]
        # if a == 0:
        #     next_s = s - self.grid_c
        # elif a == 1:
        #     next_s = s + 1
        # elif a == 2:
        #     next_s = s + self.grid_c
        # elif a == 3:
        #     next_s = s - 1

        # get rewards
        # reward = 0

        text = input("reward (1 or -1): ")  # Python 3
        reward = int(text)
        # equals = True
        # for i in range(len(a)):
        #     if a[i] != seen_a[i]:
        #         equals = False
        # if equals:
        #     reward = 1
        # else:
        #     reward = -1

        # next state
        # next_s = s 
        # next_s = int(self.transitions[s, a])

        # check if final
        # done = self.is_final(next_s)
        done = False
        
        return (next_s, reward, done)
        # return (next_s, self.rewards[coords[0], coords[1], a], done)

    def is_final(self, s):
        # coords = self.state_to_coord[s]
        # if coords[0] == 3 and coords[1] == 4:
        #     return True
        # else:
        return False


class HumaneMethodsEnvironment:
    # Gola: reach this sound target: 
    # [0.964286, 0.892857, 0.755814, 0.674419, 0.053571, 0.6875, 0.669643, 0.017857, 0.508929, 0.348837, 0.544643, 0.4375, 0.571429, 0.209302, 0.186047, 0.5625, 0.285714, 0.116279, 0.571429, 0.428571]
    def __init__(self, dimension=10, step_size=0.2):
        # self.goal = [0.965116, 0.755814, 0.581395, 0.488372, 0.383721, 0.302326, 0.244186, 0.186047, 0.151163, 0.081395]
        # self.n_dim = len(self.goal)
        self.n_dim = dimension
        self.goal = [0.5 for i in range(self.n_dim)] #[float(i%2) for i in range(self.n_dim)]
        self.n_quant = step_size
        self.n_states = self.n_dim 
        # +1/-1 for i in [1,ndim] 
        #   => (0, 1, .., ndim) = + 1 
        #   => (ndim+1, ..., 2*ndim) = - 1 
        self.n_actions = 2 * self.n_dim 

    def sample_initial_state(self):
        s = np.random.rand(self.n_dim)
        s = np.clip(s, 0., 1.)
        return np.array(s).reshape((1, self.n_dim))

    def actions(self, s):
        #coords = self.state_to_coord[s]
        #acts = np.where(self.transition[coords[0], coords[1], :] != 0)[0]
        valid_acts = []
        for i in range(self.n_dim):
            if s[0, i] + self.n_quant < 1.0:
                valid_acts.append(i)
        for i in range(self.n_dim):
            if s[0, i] - self.n_quant > 0.0:
                valid_acts.append(self.n_dim + i)
        # print(s, valid_acts)
        return valid_acts

    def set_step_size(self, step_size):
        self.n_quant = step_size

    def next_state(self, state, action):
        idx = action
        new_state = np.copy(state)
        if idx < self.n_dim:
            new_state[0, idx] += self.n_quant
        elif idx >= self.n_dim:
            new_state[0, idx - self.n_dim] -= self.n_quant

        return new_state

    def distances(self, s):
        dists = [(s[0,i] - self.goal[i])**2 for i in range(self.n_dim)]
        return dists

    def state_reward(self, s, a):
        dist = (np.sqrt(np.sum([(s[0,i] - self.goal[i])**2 for i in range(self.n_dim)]))/self.n_dim) ** 2
        # if len(np.where(s[0, :] > 1.0)[0]): 
        #     reward = abs(((1.0 - dist) ** 2)) #0.0
        #     for i in np.where(s[0, :] > 1.0)[0]:
        #         reward += 2 * (1.0 - s[0,i]) #* 1000.0
        # elif len(np.where(s[0,:] < 0.0)[0]):
        #     reward = abs(((1.0 - dist) ** 2)) #0.0
        #     for i in np.where(s[0,:] < 0.0)[0]:
        #         reward += 2*s[0,i] #* 1000.0
        # else:
        #     reward = abs(((1.0 - dist) ** 2)) #* 100.0
        reward = - (dist * 10.0)
        # print(len(np.where(s[0,:] > 1.0)[0]), s[0,:], reward)
        # print('state_rwd', a, np.argmax(a))
        next_s = self.next_state(s, a)
        # next_s = np.copy(s)
        # if a < self.n_dim:
        #     next_s[0, a] += 1/self.n_quant
        # elif a >= self.n_dim:
        #     next_s[0, a - self.n_dim] -= 1/self.n_quant
        done = self.is_final(next_s)
        #print(reward, dist, done, ['{:.3f}'.format(i) for i in s[0,:]])
        return (next_s, reward, done, dist)

    def is_final(self, s):
        dist = np.sqrt(np.sum([(s[0,i] - self.goal[i])**2 for i in range(self.n_dim)]))/self.n_dim
        if dist < 0.03:
            return True
        else:
            return False


