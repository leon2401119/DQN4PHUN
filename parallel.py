import compiler_gym
from torch.multiprocessing import Pool,cpu_count,set_start_method
import time
import random
import numpy as np
from DQN import *

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def runner(policy_net,benchmark,max_steps,epsilon,reward_spec,final_size_only):
    env = compiler_gym.make("llvm-v0",observation_space="Autophase",reward_space=reward_spec)
    env.reset(benchmark=benchmark)

    trajectory = []

    observation = env.observation["Autophase"]
    with torch.no_grad():
        for i in range(max_steps):
            #print(i)
            if random.random() > epsilon:
                q = policy_net(torch.tensor(observation,dtype=torch.float32))
                
                if i > 1 and trajectory[-1][1] == trajectory[-2][1]: # no more than 1 consecutive actions
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
        
            if done:
                break

            observation = observation_next

    trajectory[-1][3] = True  # set done to True for the last step
    
    if final_size_only:
        final_size = env.observation["ObjectTextSizeBytes"]
        env.close()
        return final_size

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
    stress_test(runner,(policy_net,"cbench-v1/patricia",EP_LENGTH,0),JOBS)

