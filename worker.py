import compiler_gym
import time
import random
import numpy as np
from multiprocessing.connection import Client,Listener
import os
import sys


def train_runner(benchmark,max_steps,reward_spec,epsilon):
    #env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space=reward_spec)
    env.reset(benchmark=benchmark)

    trajectory = []

    observation = env.observation["Autophase"]
    steps = 0
    while steps < max_steps:
        if random.random() > epsilon:
            conn = Client(('localhost', 6000), authkey=b'secret password')
            conn.send(('inference',observation))
            q = conn.recv()
            conn.close()

            if steps > 1 and trajectory[-1][1] == trajectory[-2][1]:
                # forbid more than 1 same consecutive actions
                q[trajectory[-1][1]] = -10000

            action = q.argmax().item()
            
        else:
            action = env.action_space.sample()

        try: # remember kids, compiler crashes DO occur
            observation_next,reward,done,info = env.step(action)
        except Exception as e:
            break # terminate immediately, and let the last successful action be the last step

        trajectory.append([observation,action,reward,done,observation_next])
        steps += 1

        if done or reward < -100000: # give reward a threshold if the benchmark becoming too large
            break

        observation = observation_next


    trajectory[-1][3] = True  # set done to True for the last step 
    
    #env.close()
    conn = Client(('localhost', 6000), authkey=b'secret password')
    conn.send(('enqueue',trajectory))
    conn.close()

    return steps


def test_runner(benchmark,max_steps,reward_spec,epsilon=None):
    #env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space=reward_spec)
    env.reset(benchmark=benchmark)

    trajectory = []

    observation = env.observation["Autophase"]
    steps = 0
    while steps < max_steps:

        conn = Client(('localhost', 6000), authkey=b'secret password')
        conn.send(('inference',observation))
        q = conn.recv()
        conn.close()
                
        if steps > 1 and trajectory[-1][1] == trajectory[-2][1]: # no more than 1 consecutive actions
            q[trajectory[-1][1]] = -1

        while True:
            action = q.argmax().item()

            if q[action] < 0: # no pass is predicted to improve
                break

            try:
                observation_next,reward,done,info = env.step(action)
            except Exception as e:
                return np.NINF # negative infinity reward

            if not info['action_had_no_effect']:
                break

            else:
                q[action] = -1

        if q[action] < 0:
            break # break again from inner break condition

            
        trajectory.append([observation,action,reward,done,observation_next])
        steps += 1

        if done:
            break

        observation = observation_next

    trajectory[-1][3] = True  # set done to True for the last step
    
    final_size = env.observation["ObjectTextSizeBytes"]
    #env.close()

    bench_name = benchmark.split('/')[-1]
    os.system(f'echo "{final_size}" >>logs/{bench_name}')
    os.system(f'echo "{env.commandline()}" >>logs/{bench_name}')

    return final_size


def main_loop(listener):
    running = True
    while running:
        conn = listener.accept()
        
        args = conn.recv()

        if args[0] == 'sample':
            steps = train_runner(*args[1:])
            conn.send(steps)
            conn.close()

        elif args[0] == 'eval':
            result = test_runner(*args[1:])
            conn.send(result)
            conn.close()

        elif args[0] == 'exit':
            conn.close()
            print('exiting')
            running = False

        else:
            print(f'Unsupported action {args[0]}')


def handshake(listener):
    conn = listener.accept()
    if conn.recv() == 'hi':
        conn.send('hi back')
    else:
        conn.close()
        sys.exit()

    conn.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please enter port where the worker will be listening')
        sys.exit()

    reward_spec = 'ObjectTextSizeBytes'
    env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space=reward_spec)
    
    listener = Listener(('localhost', int(sys.argv[1])), authkey=b'secret password')
    handshake(listener)
    main_loop(listener)

    JOBS = 100
    WORKERS = 10
    EP_LENGTH = 30


