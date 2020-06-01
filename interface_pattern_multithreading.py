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
# import parser


use_interaction = False
goforit = True
scrumble = False
scrumble_thresh = 5
start_new_episode = False
light_status = 0



performer_state = []

# udp message handler functions 
def handler(address, *args):
    global goforit
    global performer_state
    global scrumble
    global scrumble_thresh

    arguments = args[0][0]
    state_values = np.array(args[1:])
    
    goforit = True
    scrumble = False
    performer_state = state_values

    cum_prod = np.cumsum(performer_state)[-1]
    print("receiving values:", state_values, 'cumsum:', cum_prod)
    if cum_prod > scrumble_thresh:
        scrumble = True

mmg1_data_collector = []
# udp message handler functions 
def handler_mmg1(address, *args):
    global mmg1_data_collector
    # print('handler_mmg1', args[1])
    mmg1_data_collector.append(float(args[1]))

mmg2_data_collector = []
# udp message handler functions 
def handler_mmg2(address, *args):
    global mmg2_data_collector
    # mmg1_data_collector.append(float(args[1][0]))


# udp message handler functions 
def handler_lights(address, *args):
    global start_new_episode
    global light_status
    on = int(args[1])
    if (light_status == 1) and (not on): 
        print('start_new_episode True')
        start_new_episode = True
    light_status = on
    # print('lof', onoff, args)



def json_parser(config_file):

    if config_file == "":
        clients = [{'ip': '127.0.0.1', 'portout': 5009}]
        schedule = []
        for i in range(100):
            schedule.append({'episodes': [i], 'n_iterations': 100, 't_sleep': 0.0})
        env_config = {"step_size": 0.02}
        model = {"pre_trained_model": ""}
    
    else:
        config = json.load(open(config_file, 'r'))
        net = config['network']
        osc_server = net['server']
        osc_clients = net['clients']
        gen_config = config['need_sensors']
        env_config = config['environment']
        model_config = config['model']
        if "log" not in config.keys():
            logging = True
        else:
            logging = config["log"]
        schedule = config['schedule']
        for i in range(len(schedule)):
            if 'n_iterations' not in schedule[i].keys(): 
                schedule[i]['n_iterations'] = 100
            if 't_sleep' not in schedule[i].keys(): 
                schedule[i]['t_sleep'] = 0.0
            if 'scrumble_thresh' not in schedule[i].keys(): 
                schedule[i]['scrumble_thresh'] = 5   
    return osc_server, osc_clients, gen_config, env_config, model_config, schedule, logging


# parse input json file name and data into json file
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, default="schedule.json", help="JSON file describing leearning schedule")
args = arg_parser.parse_args()
oscserver, oscclients, gen_config, env_config, model_config, schedule, logging = json_parser(args.config)


use_interaction = gen_config

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
dispatcher_for_handler.map("/lof", handler_lights, args)
dispatcher_for_handler.map("/mmg1-features", handler_mmg1, args)
dispatcher_for_handler.map("/mmg2-features", handler_mmg2, args)


# init environment and learning algorithm
env = environment.HumaneMethodsEnvironment(**env_config)
agent = learning.Dumby(env, epsilon=0.9, gamma=0.95, algorithm='dqn', schedule=schedule)    


async def master_loop():
  
    global goforit
    global scrumble
    global scrumble_thresh
    global start_new_episode
    global mmg1_data_collector
    global mmg2_data_collector

    # # handle arguments IP, port in and out
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", 
    #     type=str, 
    #     default="", 
    #     help="JSON file describing leearning schedule")
    # args = parser.parse_args()

    # clients, model_config, schedule, logging = parse_config(args.config)

    # # OSC client
    # udp_clients = []
    # for ci, c in enumerate(clients):
    #     udp_clients.append(udp_client.SimpleUDPClient(c['ip'], c['portout']))

    # # run learning
    # env = environment.ToySynths()
    # agent = learning.Dumby(env, epsilon=1.0, gamma=0.95, algorithm='dqn', schedule=schedule)

    # if model_config['pre_trained_model'] != "":
    #     print('loading model {}'.format(model_config['pre_trained_model']))
    #     agent.load(model_config['pre_trained_model'])

    agent_act = True
    enough_data = False
    requested_size = 50
    mmg1level = 0
    mmg2level = 0
    direction = 1
    prev_reward = 0

    if model_config['training']:

        # timestamp = time.strftime("%d-%m-%y_%H-%M-%S", time.gmtime())
        print('start learning...')
        n_episodes = schedule[-1]['episodes'][-1]
        history = []

        for e in range(n_episodes):

            # print('episode #{:03d}'.format(e+1))
            print("Starting new learning loop")
            
            for ci, c in enumerate(udp_clients):
                c.send_message("/episode", e+1)

            history_e = []
            for i in range(len(schedule)):
                if e+1 >= schedule[i]['episodes'][0] and e+1 <= schedule[i]['episodes'][-1]:
                    break

            n_loops = schedule[i]['n_iterations']
            t_sleep = schedule[i]['t_sleep']
            scrumble_thresh = schedule[i]['scrumble_thresh']

            env.set_step_size(schedule[i]['environment']['step_size']) 

            # initial state
            state = env.sample_initial_state()
            tot_reward = 0
            tot_loss = 0
            time_episode = time.time()
            agent_act = True
            loop = 0
            previous_dist_mean = 0.
            direction_table = []

            while loop < n_loops:

                # if goforit:
                if agent_act:

                    #print("... episode {:03d} - loop {:05d} (time: {:04d})".format(e+1, loop+1, int(time.time() - time_episode)))
                    
                    action = agent.act(state,  env.actions(state))
                    action_ = action if action - env_config['dimension'] < 0 else action - env_config['dimension']
                    direction_ = np.sign(action - env_config['dimension'])
                    print(". Exploring parameter #{} via direction: {}".format(action_, direction_))

                    #print('env.state_reward')
                    next_state, reward, done, dist = env.state_reward(state, action)
                    st = 'getting closer' if prev_reward - reward < 0 else 'moving away'
                    print(". Received a reward of {} for that move, {}".format(reward, st))
                    prev_reward = reward

                    direction_table.append(dist)
                    if len(direction_table) > 20:
                        if previous_dist_mean > np.mean(direction_table):
                            direction = -1
                        else:
                            direction = 1
                        previous_dist_mean = np.mean(direction_table)
                        del direction_table[:10]

                    # print('env.state_reward output', next_state, reward, done, dist)

                    # for ci, c in enumerate(udp_clients):
                    #     c.send_message("/capture", 1)

                    # loop += 1
                    # goforit = False

                    if use_interaction:
                        agent_act = False
                    else:
                        agent_act = True

                if use_interaction:
                    # await asyncio.sleep(1.0)
                    if len(mmg1_data_collector) >= requested_size:
                        # print(loop, len(mmg1_data_collector))
                        goforit = True
                        mmg1mean = np.mean(mmg1_data_collector)
                        if mmg1mean  < 0.01:
                            mmg1level = 1
                        elif mmg1mean >= 0.01 and mmg1mean < 0.02:
                            mmg1level = 2
                        elif mmg1mean > 0.02 and mmg1mean < 0.04:
                            mmg1level = 3
                        elif mmg1mean >= 0.04:
                            mmg1level = 4
                        mmg2mean = np.mean(mmg2_data_collector)
                        if mmg2mean  < 0.01:
                            mmg2level = 1
                        if mmg2mean >= 0.01 and mmg2mean < 0.02:
                            mmg2level = 2
                        if mmg2mean >= 0.02 and mmg2mean < 0.04:
                            mmg2level = 3
                        if mmg2mean >= 0.04:
                            mmg2level = 4
                        mmg1_data_collector = []
                        mmg2_data_collector = []
                else:
                    goforit = True

                # print(goforit)
                if goforit:

                    if mmg1level >= 3:
                        print('     mmg1level:', mmg1level, mmg1mean)
                        next_state = env.sample_initial_state()

                    next_state = np.reshape(next_state, [1, agent.get_state_size()])

                    differential = np.sum(np.power(next_state - np.reshape(state, [1, agent.get_state_size()]), 2)) / agent.get_state_size()

                    # print(scrumble, next_state, action, state)

                    # Remember the previous state, action, reward, and done
                    # print("remember", loop)
                    agent.remember(state, action, reward, next_state, done)

                    state = next_state

                    distances = env.distances(state)
                    distance_levels = []
                    for i in range(len(distances)):
                        if distances[i] < 0.025: 
                            distance_levels.append(2)
                        elif distances[i] < 0.1:
                            distance_levels.append(1)
                        else:
                            distance_levels.append(0)
                    # print(distance_levels)
                    # loss = agent.update(state, action, next_state, reward, done, loop)
                    
                    # history_e.append({'state': state, 
                    #                   'action': action, 
                    #                   'next_state': next_state, 
                    #                   'reward': reward, 
                    #                   'done': done, 
                    #                   'loop': loop}) 
                    
                    
                    # tot_loss += loss
                    # time.sleep(t_sleep)

                    # send through udp
                    for ci, c in enumerate(udp_clients):
                        c.send_message("/duration", t_sleep)
                        c.send_message("/values", next_state[0,:])
                        c.send_message("/status", [e+1, loop+1, t_sleep])
                        c.send_message("/rewards", [reward, dist])
                        c.send_message("/direction", direction)
                        c.send_message("/distances", distance_levels)

                    if use_interaction:
                        goforit = False
                    else:
                        goforit = True

                    agent_act = True
                    tot_reward += reward
                    loop += 1

                # await asyncio.sleep(0.0)

                if not use_interaction:
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.0)

                if start_new_episode:
                    start_new_episode = False
                    break

            # tot_loss = agent.replay()   
            # agent.forget()

            # history.append({'history_episode':history_e, 'loss': tot_loss})

            # print('episode #{:02d} in {}, reward: {}, loss: {}'.format(e+1, time.time() - time_episode, tot_reward, tot_loss), 'step_size:', schedule[i]['environment']['step_size'], agent.get_epsilon())

        # if logging:
        #     pickle.dump(history, open('history_{}.pkl'.format(timestamp), 'wb'))
        #     agent.save(name="model_{}.h5".format(timestamp))

    elif model_config['inference']:
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

async def init_main():
    print('serving on:', server_ip, server_port)
    server = AsyncIOOSCUDPServer((server_ip, server_port), 
                                 dispatcher_for_handler, 
                                 asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await master_loop()  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

asyncio.run(init_main())



