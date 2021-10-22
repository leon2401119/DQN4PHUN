import compiler_gym
import numpy as np
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from collections import deque
from DQN import *
from util import replay_buffer
from torch.multiprocessing import Pool,cpu_count,set_start_method
from parallel import runner
import os


try:
    set_start_method('spawn')
    print('set start method = spawn')
except RuntimeError:
    pass

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_available_datasets(env):
    for dataset in env.datasets:
        print(dataset.name)


def get_dataset(env,dataset="benchmark://cbench-v1"):
    uris = []

    for uri in env.datasets[dataset].benchmark_uris():
        # filter benchmarks that take too long to build
        env.reset(benchmark=uri)
        uris.append(uri)
        if env.observation["IsBuildable"] and env.observation["Buildtime"].item() < 1:
            uris.append(uri)

    return uris


def split_train_and_test(data,test_ratio=0.25):
    random.shuffle(data)
    split = len(data) - int(len(data)*test_ratio)
    return data[:split],data[split:]



def sample_traj(episodes=5):
    global steps

    pool = Pool(processes=6) # setting >6 will trigger CUDA OOM error
    args = []
    epsilon = E_END + (E_START - E_END) * math.exp(-1. * steps / E_DECAY)
    #epsilon = 0
    for _ in range(episodes):
        args.append((policy_net,train_corpus[random.randint(0,len(train_corpus)-1)],EP_MAX_STEPS,epsilon,'ObjectTextSizeBytes',False))

    trajs = pool.starmap(runner,[arg for arg in args])
    for traj in trajs:
        for step in traj:
            memory.insert(step)
    
    steps += (EP_MAX_STEPS*episodes)

    #for _ in range(episodes):
        # TODO : this is NOT a scalable approach
        #env.reset(benchmark=train_corpus[random.randint(0,len(train_corpus)-1)])
        #env.reset(benchmark="cbench-v1/patricia")
        #print(env.benchmark)

        #total_reward = 0

        #for i in range(EP_MAX_STEPS):
            #print(i)
            #observation = env.observation["Inst2vec"]
            #observation = env.observation["Autophase"]
            
            #global steps
            #epsilon = E_END + (E_START - E_END) * math.exp(-1. * steps / E_DECAY)
            #steps += 1

            #if random.random() < epsilon:
            #    action = env.action_space.sample()
            #else:
            #    action = policy_net(torch.tensor([observation],dtype=torch.float32)).argmax().item()
                #print(type(action.)) # action is int

            #_, reward, done, _ = env.step(action)
            #observation_next = env.observation["Inst2vec"]
            #observation_next = env.observation["Autophase"]
            #total_reward += reward

            #memory.insert([observation,action,reward,observation_next])

            #if done:
            #    break

        #global max_seen_total_reward
        #if total_reward > max_seen_total_reward:
        #    max_seen_total_reward = total_reward


def get_base_absolute_size(corpus,option='Oz'):
    assert option == 'Oz' or option == 'O3' or option == 'O0', 'unrecognized option'
    obsv_space = 'ObjectTextSize' + option
    
    env = compiler_gym.make("llvm-v0")
    size = []
    for benchmark in corpus:
        env.reset(benchmark=benchmark)
        size.append(env.observation[obsv_space])

    env.close()
    return np.array(size)

def get_oz_size_reduction(corpus):
    return get_base_absolute_size(corpus,'Oz')/get_base_absolute_size(corpus,'O0')


def validate(corpus):
    pool = Pool(processes=6)
    args = []

    for benchmark in corpus:
        args.append((policy_net,benchmark,EP_MAX_STEPS,0,'ObjectTextSizeBytes',True))

    sizes = pool.starmap(runner,[arg for arg in args])
    #reward_list = []
    #for traj in trajs:
    #    reward_list.append(0)
    #    for step in traj:
    #        reward_list[-1] += step[2]

    return sizes


    #reward_list = []

    #for benchmark in corpus:
    #    reward,_,_ = validate_one(benchmark)
    #    reward_list.append(reward)

    #mean_reward = np.array(reward_list).mean()
    
    #return mean_reward


def validate_one(benchmark):
    #traj = runner(policy_net,benchmark,EP_MAX_STEPS,0).detach().cpu().numpy()
    #zipped_traj = numpy.column_stack(*traj)
    #return zipped_traj[2],zipped_traj[1],zipped_traj[


    env_val = compiler_gym.make("llvm-v0",reward_space="ObjectTextSizeBytes")
    env_val.reset(benchmark=benchmark)
    actions = []
    reward = 0
    with torch.no_grad():
        for i in range(EP_MAX_STEPS):
            q = policy_net(torch.tensor([env_val.observation["Autophase"]],dtype=torch.float32))

            if i == 0:
                q_temp = q


            if len(actions) > 1 and actions[-1] == actions[-2]: #never take >2 same consecutive passes
                q[0][actions[-1]] = -1

            while True:
                if q.max().item()<0:
                    return reward,actions,q_temp.detach().cpu().numpy()
                
                action = q.argmax().item()
                _,r,done,info = env_val.step(action)
                
                if not info['action_had_no_effect']:
                    reward += r
                    actions.append(action)
                    break

                else: # action has no effect
                    q[0][action] = -1 # render unusable
            

    return reward,actions,q_temp.detach().cpu().numpy()


def train():
    global updates

    while(len(memory) >= BATCH_SIZE):
        batch = memory.get_batch(BATCH_SIZE)
        s0, s1, r, a, done = [],[],[],[],[]
        for i in range(BATCH_SIZE):
            #s0.append(torch.tensor(batch[i][0],dtype=torch.float32))
            #s1.append(torch.tensor(batch[i][3],dtype=torch.float32))
            s0.append(batch[i][0])
            s1.append(batch[i][4])
            done.append(batch[i][3])
            r.append([batch[i][2]])
            a.append([batch[i][1]])

        #s0 = pad_sequence(s0,batch_first=True) # s0.shape = (batch,1489+-,200)
        s0 = torch.tensor(s0,dtype=torch.float32).to(device)
        s1 = torch.tensor(s1,dtype=torch.float32).to(device)
        #s1 = pad_sequence(s1,batch_first=True)
        r = torch.tensor(r).to(device)
        a = torch.tensor(a).to(device)

        start = time.time()
        # policy_net(s0) has output shape of (batch,num_actions)
        Q_s0 = torch.gather(policy_net(s0),1,a)
        Q_s1 = target_net(s1).max(dim=1,keepdim=True)[0].detach() # max returns (max,argmax)
        Q_target = Q_s1 + GAMMA*r
        for j,mask in enumerate(done):
            if mask:
                Q_target[j] = r[j]

        end = time.time()
        os.system(f'echo "cal Qs : {end-start}" >>profiling')

        #print(Q_s0, Q_s1, GAMMA*r)
        #print(Q_s0 - Q_s1)
        #assert True == False

        start = time.time()
        #criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        #loss = criterion(Q_s0, Q_s1 + GAMMA*r)
        loss = criterion(Q_s0, Q_target)
        end = time.time()
        os.system(f'echo "cal loss : {end-start}" >>profiling')


        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        end = time.time()
        os.system(f'echo "backward + update : {end-start}" >>profiling')

        updates += 1

        os.system(f'echo "{loss}" >>loss_history')
        #if not updates % 100:
        #    print(f'UPDATE {updates}, LOSS = {loss}')

        if not updates % 100:
            #print(f'{updates} : Mean Reward = ',end='')
            #print(f'{validate(train_corpus)}/{validate(test_corpus)}')
            #r,a,q = validate_one(test_corpus[0])
            #print(f'{updates} : {r}, {a}, {q}')
            #print(f'{updates} : {r},{a}')
            reduction = validate(test_corpus)/test_o0_absolute_size
            print(f'{reduction.mean()}/{reduction.std()}')


        if not updates % TARGET_UPDATE:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == '__main__':
    os.system(f'rm -f profiling')
    os.system(f'rm -f loss_history')

    #env = compiler_gym.make("llvm-v0",benchmark="cbench-v1/qsort")
    env = compiler_gym.make("llvm-v0",observation_space="IrSha1",reward_space="ObjectTextSizeNorm")
    env.reset()

    #corpus = get_dataset(env,"benchmark://blas-v0")
    corpus = get_dataset(env)
    train_corpus, test_corpus = split_train_and_test(corpus)
    print(f'Corpus Size : {len(corpus)} ({len(train_corpus)}/{len(test_corpus)})')

    #print(get_base_absolute_size(train_corpus,'O0'))
    #print(get_base_absolute_size(train_corpus,'Oz'))
    #print(get_base_absolute_size(train_corpus,'Oz')/get_base_absolute_size(train_corpus,'O0'))
    #assert True==False
    test_o0_absolute_size = get_base_absolute_size(test_corpus,'O0')
    test_oz_size_reduction = get_oz_size_reduction(test_corpus)
    print(f'Oz : {test_oz_size_reduction.mean()}/{test_oz_size_reduction.std()}')
    

    BATCH_SIZE = 1
    GAMMA = 0.999
    #GAMMA = 0.5
    E_START = 0.99
    E_END = 0.5
    #E_DECAY = 2000
    E_DECAY = 200000
    #TARGET_UPDATE = 10000
    TARGET_UPDATE = 500
    EP_MAX_STEPS = 50
    #EP_MAX_STEPS = 1
    LEARNING_RATE = 0.0005


    #policy_net = DQN(len(env.observation["Inst2vec"][0]),len(env.action_space.names)).to(device)
    #target_net = DQN(len(env.observation["Inst2vec"][0]),len(env.action_space.names)).to(device)
    policy_net = DQN(len(env.observation["Autophase"]),len(env.action_space.names)).to(device)
    target_net = DQN(len(env.observation["Autophase"]),len(env.action_space.names)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #r,a,_ = validate_one(test_corpus[0])
    #print(r,a)
    #assert True == False

    policy_net.share_memory()

    optimizer = optim.Adam(policy_net.parameters(),lr=LEARNING_RATE)
    memory = replay_buffer()

    steps = 0
    max_seen_total_reward = -10000


    reduction = validate(test_corpus)/test_o0_absolute_size 
    print(f'Init : {reduction.mean()}/{reduction.std()}')

    updates = 0
    while updates < 1000000:
        sample_traj(20)
        train()

    env.close()


