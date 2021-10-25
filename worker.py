import compiler_gym
import time
import random
import numpy as np
from multiprocessing.connection import Client,Listener

import sys

def runner(benchmark,max_steps,epsilon,reward_spec,final_size_only):
    env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space=reward_spec)
    env.reset(benchmark=benchmark)

    trajectory = []

    observation = env.observation["Autophase"]
    steps = 0
    while steps < max_steps:

    #for i in range(max_steps):
        #print(i)
        if random.random() > epsilon:
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

                observation_next,reward,done,info = env.step(action)
                    
                if not info['action_had_no_effect']:
                    break

                else:
                    q[action] = -1

            if q[action] < 0:
                break # break again from inner break condition

        else:
            action = env.action_space.sample()
            observation_next,reward,done,info = env.step(action)
            
        trajectory.append([observation,action,reward,done,observation_next])
        steps += 1

        if done:
            break

        observation = observation_next

    trajectory[-1][3] = True  # set done to True for the last step
    
    if final_size_only:
        final_size = env.observation["ObjectTextSizeBytes"]
        env.close()
        return final_size

    env.close()

    conn = Client(('localhost', 6000), authkey=b'secret password')
    conn.send(('enqueue',trajectory))
    conn.close()

    return steps


def main_loop(listener):
    running = True
    while running:
        conn = listener.accept()
        
        args = conn.recv()

        if args[0] == 'sample':
            steps = runner(*args[1:])
            conn.send(steps)
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

    listener = Listener(('localhost', int(sys.argv[1])), authkey=b'secret password')
    handshake(listener)
    main_loop(listener)

    JOBS = 100
    WORKERS = 10
    EP_LENGTH = 30

