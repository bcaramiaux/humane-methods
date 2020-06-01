#!/usr/bin/env python

"""
Copyright 2019 Baptiste Caramiaux 
"""

import argparse
import random
import time
import math
import numpy as np
import pickle
import json
import asyncio
import matplotlib.pylab as plt

from pythonosc import udp_client
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher

import environment
import learning
import parser



# data containers
mmg1_data_container = []
mmg2_data_container = []


# async def run():
#     time.sleep(1.0)
    

goforit = True
performer_state = []

# udp message handler functions 
def handler(address, *args):
    global goforit
    global performer_state
    arguments = args[0][0]
    state_values = np.array(args[1:])
    print("state_values:", state_values)
    goforit = True
    performer_state = state_values
    # await run()


# parse input json file name and data into json file
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, default="schedule.json", help="JSON file describing leearning schedule")
args = arg_parser.parse_args()
oscserver, oscclients, sigs, model_config, schedule, logging = parser.json_parser(args.config)


# OSC 
# clients
udp_clients = []
for ci, c in enumerate(oscclients):
    print(c['ip'], c['portout'])
    udp_clients.append(udp_client.SimpleUDPClient(c['ip'], c['portout']))
# hanlders
server_ip = oscserver['ip']
server_port = oscserver['port']
dispatcher_for_handler = Dispatcher()
dispatcher_for_handler.map("/mmg-levels", handler, args)


# init environment and learning algorithm
env = environment.HumaneMethodsEnvironment()
agent = learning.Dumby(env, epsilon=0.2, gamma=0.95, algorithm='sarsa', schedule=schedule)    


# def run_one_step(env, agent, state, loop):
#     # agent chooses an action = choreographic scene
#     print('actions', env.actions(state))
#     action = agent.act(state,  env.actions(state))

    
#     return next_state, reward


async def master_loop():

    global goforit
    global performer_state

    n_episodes = schedule[-1]['episodes'][-1]

    for e in range(n_episodes):        

        for i in range(len(schedule)):
            if e+1 >= schedule[i]['episodes'][0] and e+1 <= schedule[i]['episodes'][-1]:
                break
        n_loops = schedule[i]['n_iterations']

        # initial state
        current_state = env.sample_initial_state()

        for ci, c in enumerate(udp_clients):
            c.send_message("/capture", 1)

        loop = 0
        while loop < n_loops:

            if goforit:
                # print('actions', env.actions(current_state))
                action = agent.act(current_state,  env.actions(current_state))  # scene to peform 
                
                print(current_state, env.get_state_name(action))
                for ci, c in enumerate(udp_clients):
                    c.send_message("/capture", 1)

                loop += 1
                goforit = False

            await asyncio.sleep(0.0)

            if goforit:
                if len(performer_state) == 0:
                    print('BUUUUG')
                else:
                    next_state_vec = np.copy(performer_state)
                    next_state = env.coord_to_state(next_state_vec)
                    next_state, reward, done = env.state_reward(current_state, action, next_state)
                    print(current_state, next_state, reward, action)
                    agent.update(current_state, action, next_state, reward, done, loop)
                    current_state = next_state
                    goforit = True
                    time.sleep(1.0)
            
        print('Just finished episode {} with loop {}'.format(e, loop))



async def init_main():
    print(server_ip, server_port)
    server = AsyncIOOSCUDPServer((server_ip, server_port), dispatcher_for_handler, asyncio.get_event_loop())
    # print("Serving on {}".format(server.server_address))
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await master_loop()  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

asyncio.run(init_main())

