import numpy as np
import gym
import gym_target
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint, FileLogger


ENV_NAME = 'target-v0'
ENV_NAME = 'CartPole-v0'
# ENV_NAME = 'target-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=10,
               target_model_update=1e-2, 
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

if os.path.isfile('dqn_{}_weights.h5f'.format(ENV_NAME)):
    print('loading')
    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    callbacks = build_callbacks(ENV_NAME)
    dqn.test(env, nb_episodes=2, verbose=1, nb_max_episode_steps=1000, visualize=False)
    
    #print(history.history)
else:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    callbacks = build_callbacks(ENV_NAME)
    dqn.fit(env, nb_steps=1000, visualize=False, verbose=2, callbacks=callbacks)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)