import argparse
import random
import time
import math
import numpy as np
import pickle
import json

import environment
import learning

from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))


def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass


def handle_outputs(unused_addr, args, vals):
    print("[{0}] ~ {1}".format(args[0], vals))

def handle_inputs(unused_addr, args, vals):
    print("[{0}] ~ {1}".format(args[0], vals))
    # args[1] = client
    #args[1].send_message("/values", vals)

def handler(address, *args) -> None:
    print("{}".format(args))
    args[0][0].send_message("/values", next_state[0,:])

def parse_config(config_file):
    if config_file == "":
        clients = [{'ip': '127.0.0.1', 'portout': 5009}]
        schedule = []
        for i in range(100):
            schedule.append({'episodes': [i], 'n_iterations': 100, 't_sleep': 0.0})
        env_config = {"step_size": 0.02}
        model = {"pre_trained_model": ""}
    else:
        config = json.load(open(config_file, 'r'))
        clients = config['clients']
        schedule = config['schedule']
        model_config = config['model']
        if "log" not in config.keys():
            logging = True
        else:
            logging = config["log"]
    return clients, model_config, schedule, logging



if __name__ == "__main__":
  
    # handle arguments IP, port in and out
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
        type=str, 
        default="", 
        help="JSON file describing leearning schedule")
    args = parser.parse_args()

    clients, model_config, schedule, logging = parse_config(args.config)

    # OSC client
    udp_clients = []
    for ci, c in enumerate(clients):
        udp_clients.append(udp_client.SimpleUDPClient(c['ip'], c['portout']))

    # run learning
    env = environment.ToySynths()
    agent = learning.Dumby(env, epsilon=1.0, gamma=0.95, algorithm='dqn', schedule=schedule)

    if model_config['pre_trained_model'] != "":
        print('loading model {}'.format(model_config['pre_trained_model']))
        agent.load(model_config['pre_trained_model'])

    if model_config['training']:
        timestamp = time.strftime("%d-%m-%y_%H-%M-%S", time.gmtime())
        print('start learning...')
        n_episodes = schedule[-1]['episodes'][-1]
        history = []
        for e in range(n_episodes):

            history_e = []
            for i in range(len(schedule)):
                if e+1 >= schedule[i]['episodes'][0] and e+1 <= schedule[i]['episodes'][-1]:
                    break
            n_loops = schedule[i]['n_iterations']
            t_sleep = schedule[i]['t_sleep']

            env = environment.ToySynths(step_size=schedule[i]['environment']['step_size'])

            # print(i, n_loops, t_sleep, schedule[i]['environment']['step_size'], agent.get_epsilon())

            # initial state
            state = env.sample_initial_state()
            tot_reward = 0
            tot_loss = 0
            time_episode = time.time()
            for loop in range(n_loops):

                action = agent.act(state,  env.actions(state))
                next_state, reward, done, dist = env.state_reward(state, action)
                # next_state = np.clip(next_state, 0.0, 1.0)
                
                # check if final state otherwise update
                if done:
                    print("episode: {}/{}, score: {}, rwd:{}, state: {}".format(e, n_episodes, loop, reward, next_state))
                    break

                next_state = np.reshape(next_state, [1, agent.get_state_size()])

                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)

                state = next_state

                # loss = agent.update(state, action, next_state, reward, done, loop)
                
                history_e.append({'state': state, 
                                  'action': action, 
                                  'next_state': next_state, 
                                  'reward': reward, 
                                  'done': done, 
                                  'loop': loop}) 
                
                tot_reward += reward
                # tot_loss += loss
                time.sleep(t_sleep)

                # send through udp
                for ci, c in enumerate(udp_clients):
                    c.send_message("/duration", t_sleep)
                    c.send_message("/values", next_state[0,:])
                    c.send_message("/status", [e+1, loop+1, t_sleep])
                    c.send_message("/rewards", [reward, dist])

            tot_loss = agent.replay()

            # agent.forget()

            history.append({'history_episode':history_e, 'loss': tot_loss})

            
            print('episode #{:02d} in {}, reward: {}, loss: {}'.format(e+1, time.time() - time_episode, tot_reward, tot_loss), 'step_size:', schedule[i]['environment']['step_size'], agent.get_epsilon())

        if logging:
            pickle.dump(history, open('history_{}.pkl'.format(timestamp), 'wb'))
            agent.save(name="model_{}.h5".format(timestamp))

    if model_config['inference']:
        state = env.sample_initial_state()
        counter = 0
        while 1:
            next_state = agent.infer(state, env)
            state = next_state
            # print(state)
            for ci, c in enumerate(udp_clients):
                c.send_message("/values", state[0,:])
            time.sleep(1.0)
            if counter % 100 == 0:
                state = env.sample_initial_state()
                counter = 0
            counter += 1


