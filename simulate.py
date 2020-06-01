import argparse
import random
import time
import math
import numpy as np
import pickle
import argparse

# import environment
# import learning

from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

# parser = argparse.ArgumentParser()
# parser.add_argument("--loop", type=int, default=0, help="loop #")
# args = parser.parse_args()

# # OSC client
# client = udp_client.SimpleUDPClient("127.0.0.1", 5010)

# history = pickle.load(open('history.pkl', 'rb'))
# keys = sorted([int(k) for k in history.keys()])

# episode = args.loop
# for k in keys:
#     # print(history['{}'.format(k)][-1])
#     client.send_message("/values", history['{}'.format(k)][episode-1][0,:])
#     client.send_message("/loop", k+1)
#     time.sleep(0.1)

def parse_config(config_file):
    if config_file == "":
        clients = [{'ip': '127.0.0.1', 'portout': 5009}]
        schedule = []
        for i in range(100):
            schedule.append({'episodes': [i], 'n_iterations': 100, 't_sleep': 0.0})
    else:
        config = json.load(open(config_file, 'r'))
        print(config)
        clients = config['clients']
        schedule = config['schedule']
    return clients, schedule



if __name__ == "__main__":
  
    # handle arguments IP, port in and out
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
        type=str, 
        default="", 
        help="JSON file describing leearning schedule")
    args = parser.parse_args()

    clients, schedule = parse_config(args.config)

    agent = learning.Dumby(env, epsilon=0.90, algorithm='dqn', schedule=schedule)
    agent.load('model_27-05-19_11-23-04.h5')

    state = env.sample_initial_state()
    while 1:
        next_state = agent.infer(state)
        state = next_state
        

