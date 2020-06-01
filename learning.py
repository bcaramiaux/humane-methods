import numpy as np
import random 
import bisect
import environment
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

import time


def sample_from_distribution(distribution):
    total = np.sum(distribution)
    cdf = []
    cumsum = 0
    for w in distribution:
        cumsum += w
        result.append(cumsum / total)
    x = random.random()
    idx = bisect.bisect(cdf, x)
    return idx

def epsilon_greedy_selection(q, actions, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        # exploration
        return np.random.choice(actions)
    else:
        # exploitation
        arg = np.argsort(q[actions])[::-1]
        n_tied = sum(np.isclose(q[actions], q[actions][arg[0]]))
        return actions[np.random.choice(arg[0:n_tied])]





class Dumby():
    def __init__(self, env, epsilon=0.3, gamma=0.75, algorithm='dqn', schedule={}):
        self.state_size = env.n_states
        self.action_size = env.n_actions
        self.batch_size = 32
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.algorithm = algorithm
        self.schedule = schedule
        self.in_between_training_steps = self.batch_size
        if self.algorithm=='dqn':
            self.memory = deque(maxlen=2000)
            self.target_model = self._build_model()
        elif self.algorithm =='sarsa':
            self.alpha = 0.1
            self.q = np.zeros((self.state_size, self.action_size))
            self.q.fill(float('-inf'))
            for s in range(self.state_size):
                actions = env.actions(s)
                for a in actions:
                    self.q[s, a] = 0

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        l2_reg = 0.00001
        model = Sequential()
        # model.add(Dense(10, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
        # model.add(Dropout(0.1))
        # model.add(Dense(16, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
        # model.add(Dropout(0.1))
        model.add(Dense(24, activation='relu', input_dim=self.state_size)) #, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation_regularizer=l2(l2_reg)))
        model.add(Dropout(0.01))
        model.add(Dense(24, activation='relu')) #, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation_regularizer=l2(l2_reg)))
        model.add(Dropout(0.01))
        # model.add(Dropout(0.1))
        # model.add(Dense(30, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
        # model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', 
                      optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss=self._huber_loss, 
        #               optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.batch_size and self.in_between_training_steps >= self.batch_size:
            # print(' replay')
            print('[!] Fitting model with replay')
            loss = self.replay()
            self.in_between_training_steps = 0
            
        self.in_between_training_steps += 1

    # def forget(self):
    #     del self.memory
    #     self.memory = deque(maxlen=2000)

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(actions)
            # return random.randrange(self.action_size)
        if self.algorithm=='dqn':
            act_values = self.target_model.predict(state)
            # if np.argmax(act_values[0]) not in actions:
            #     act_ = np.random.choice(actions)
            #     print('random action', act_)
            #     return act_
            # else:
            #     # print(['{:.3f}'.format(si) for si in state[0,:]], ['{:.3f}'.format(si) for si in act_values[0,:]])
            #     print('predicted action', np.argmax(act_values[0]))
            return np.argmax(act_values[0])  # returns action
        
        elif self.algorithm == 'sarsa':
            
            q_ = self.q[state]
            arg = np.argsort(q_[actions])[::-1]
            n_tied = sum(np.isclose(q_[actions], q_[actions][arg[0]]))

            return actions[np.random.choice(arg[0:n_tied])]

    def replay(self):
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
        #     target_f = self.target_model.predict(state)
        #     target_f[0][action] = target
        #     self.target_model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #minibatch = random.sample(self.memory, batch_size)
        # minibatch = self.memory
        losses = []
        #print(len(self.memory), len(self.memory[0]))
        # minibatch = self.memory #random.sample(self.memory, batch_size)
        #print(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, self.batch_size)
        counter_ = 1
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            # print(state, target_f, reward, self.gamma * np.amax(self.target_model.predict(next_state)[0]), self.target_model.predict(state))
            history = self.target_model.fit(state, target_f, epochs=1, verbose=0)
            # target = self.target_model.predict(state)
            # if done:
            #     target[0][action] = reward
            # else:
            #     # a = self.target_model.predict(next_state)[0]
            #     t = self.target_model.predict(next_state)[0]
            #     target[0][action] = reward + self.gamma * np.argmax(t)
            #     # print('log:', action, reward, np.argmax(t), reward + self.gamma * np.argmax(t))
            #     # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            # #print(state, target)
            # history = self.target_model.fit(state, target, epochs=1, verbose=0)
            # print('loss:', history.history['loss'])
            losses.append(history.history['loss'])
            print('[-] Fitting loss instance #{} in minibatch: {}'.format(counter_, history.history['loss']))
            counter_ += 1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(losses)

    # def clear_memory(self):
    #     self.memory = deque(maxlen=2000)

    def get_state_size(self):
        return self.state_size

    def get_epsilon(self):
        return self.epsilon

    def update(self, state, action, next_state, reward, done, time):
        loss = 0.0
        if self.algorithm=='dqn':
            next_state = np.reshape(next_state, [1, self.state_size])
            next_state = np.clip(next_state, 0.0, 1.0)
            self.remember(state, action, reward, next_state, done)
            #print(state, action, reward, next_state, done)
            batch_size = self.batch_size
            # print('here')
            if len(self.memory) >= batch_size:
                # print(' replay')
                loss = self.replay(batch_size)
                self.forget()
        elif self.algorithm=='sarsa':

            self.q[state, action] = self.q[state, action] \
                                            + self.alpha * (reward + self.gamma * max(self.q[next_state]) \
                                            - self.q[state, action]) #- time * 100.0
            np.savetxt('q_matrix.txt', self.q)

        return loss

    def load(self, name):
        self.target_model.load_weights(name)
        # self.target_model = load_model(name)

    def save(self, name=""):
        self.target_model.save_weights(name)
        # self.target_model.save(name)

    def infer(self, state, env):
        action_pred = self.target_model.predict(state)
        # print(action_pred)
        action = np.argmax(action_pred[0])
        next_state = env.next_state(state, action)
        next_state = np.clip(next_state, 0., 1.)
        return next_state


# def sarsa(env, n_episodes, alpha=0.1, gamma=0.9, epsilon=0.1, quickest=0.1):
#     n_states, n_actions = env.n_states, env.n_actions
#     q = np.zeros((n_states, n_actions))
#     q.fill(float('-inf'))
#     for s in range(n_states):
#         actions = env.actions(s)
#         for a in actions:
#             q[s, a] = 0
#     results = []
#     for e in range(n_episodes):
#         print('episode #{:02d}'.format(e))
#         s = env.sample_initial_state()
#         a = epsilon_greedy_selection(q[s], env.actions(s), epsilon)
#         results_e = [s]
#         counting_steps = 0
#         while not env.is_final(s):
#             next_s, r = env.state_reward(s, a)
#             next_a = epsilon_greedy_selection(q[next_s], env.actions(next_s), epsilon)
#             q[s, a] = q[s, a] + alpha * (r + gamma * q[next_s, next_a] - q[s, a]) - quickest * counting_steps
#             counting_steps += 1
#             s = next_s
#             a = next_a
#             results_e.append(next_s)
#         results.append(results_e)
#     pickle.dump(results, open('results.pkl', 'wb'))
#     print(results)
#     pi = q.argmax(axis=1)
#     v = q.max(axis=1)
#     return pi, v


def run():
    env = environment.ToySynths()
    algorithm = 'dqn'
    agent = Dumby(env, epsilon=0.1, algorithm=algorithm)
    n_episodes = 50
    history = []
    for e in range(n_episodes):
        print('episode #{:02d}'.format(e+1))
        # initial state
        state = env.sample_initial_state()

        results_e = [state]

        for time in range(300):
            print(state, state.shape)
            # get current action, compute next_state and reward
            action = agent.act(state,  env.actions(state))
            next_state, reward, done = env.state_reward(state, action)
            results_e.append(next_state)
            # check if final state otherwise update
            if done:
                print("episode: {}/{}, score: {}, state: {}".format(e, n_episodes, time, next_state))
                break
            agent.update(state, action, next_state, reward, done, time)
            state = next_state
            
        #print(results_e)
        history.append(results_e)
    print(history)

if __name__ == "__main__":
    run()


