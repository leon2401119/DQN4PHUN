import compiler_gym
from torch.multiprocessing import Pool,cpu_count
import time
import random
import numpy as np
from DQN import *


def runner(benchmark,max_steps,epsilon):
    env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space="ObjectTextSizeBytes")
    env.reset(benchmark=benchmark)

    trajectory = []

    observation = env.observation["Autophase"]
    for _ in range(max_steps):
        if random.random() > epsilon:
            action = policy_net(torch.tensor(observation,dtype=torch.float32)).argmax().item()
        else:
            action = env.action_space.sample()

        observation_next,reward,done,_ = env.step(action)
        trajectory.append([observation,action,reward,observation_next])
        
        if done:
            break

        observation = observation_next

    env.close()
    return trajectory


def stress_test(kernel,args,repeat):

    start = time.time()

    results = []
    for _ in range(repeat):
        results.append(kernel(*args))

    end = time.time()
    print('No parallelization : ' + str(end - start))

    for w in range(2,cpu_count()+5):
        start = time.time()
        pool = Pool(processes=w)
        trajectories_prl = pool.starmap(kernel,[args for _ in range(repeat)])
        end = time.time()
        print('Spawning ' + str(w) + ' workers : ' + str(end - start))


if __name__ == '__main__':
    print('Core count on this CPU : ' + str(cpu_count()))

    env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space="ObjectTextSizeBytes")
    env.reset()
    policy_net = DQN(len(env.observation["Autophase"]),len(env.action_space.names)).to(device)
    policy_net.share_memory()
    env.close()

    JOBS = 100
    WORKERS = 10
    EP_LENGTH = 30


    #run_test(JOBS,WORKERS,ACTION_QUEUE)
    stress_test(runner,("cbench-v1/patricia",EP_LENGTH,1),JOBS)

