import compiler_gym
import numpy as np
import math
import random
import time
from multiprocessing import set_start_method
#try:
#    set_start_method('spawn')
#except RuntimeError:
#    pass
from multiprocessing import Pool,cpu_count
from parallel import runner
import os
from multiprocessing.connection import Client

def print_available_datasets(env):
    for dataset in env.datasets:
        print(dataset.name)


def get_dataset(env,dataset="benchmark://cbench-v1"):
    uris = []

    for uri in env.datasets[dataset].benchmark_uris():
        # filter benchmarks that take too long to build
        env.reset(benchmark=uri)
        #uris.append(uri)
        if env.observation["IsBuildable"] and env.observation["Buildtime"].item() < 1:
            uris.append(uri)

    return uris


def split_train_and_test(data,test_ratio=0.25):
    random.shuffle(data)
    split = len(data) - int(len(data)*test_ratio)
    return data[:split],data[split:]



def sample_traj(episodes=5):

    #pool = Pool(processes=cpu_count()-1) # setting >6 will trigger CUDA OOM error
    pool = Pool(processes = cpu_count() + 20) # use more process to hide inference latency?
    args = []
    epsilon = E_END + (E_START - E_END) * math.exp(-1. * steps / E_DECAY)
    for _ in range(episodes):
        args.append((train_corpus[random.randint(0,len(train_corpus)-1)],EP_MAX_STEPS,epsilon,'ObjectTextSizeBytes',False))

    pool.starmap_async(runner,[arg for arg in args])
    pool.wait(timeout=100)

    if not pool.successful():
        pool.terminate()

    pool.close()
    pool.join()

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

    VAL_MAX_STEPS = 5
    for benchmark in corpus:
        args.append((benchmark,VAL_MAX_STEPS,0,'ObjectTextSizeBytes',True))

    sizes = pool.starmap(runner,[arg for arg in args])

    pool.close()
    pool.join()
    return sizes


def train():

    conn = Client(('localhost', 6000), authkey=b'secret password')
    #conn.send(('train',pickle.dumps(replay_memory),BATCH_SIZE,GAMMA,TARGET_UPDATE))
    conn.send(('train',BATCH_SIZE,GAMMA,TARGET_UPDATE))
    _,_ = conn.recv()
    

    reduction = validate(test_corpus)/test_o0_absolute_size
    print(f'{reduction.mean()}/{reduction.std()}')


def main(ep):
    global steps
    pool = Pool(processes = cpu_count() + 20)

    EPISODE_PER_EVAL = 100
    counter = 0

    while counter < ep:
        args = []
        epsilon = E_END + (E_START - E_END) * math.exp(-1. * steps / E_DECAY)
        for _ in range(EPISODE_PER_EVAL):
            args.append((train_corpus[random.randint(0,len(train_corpus)-1)],EP_MAX_STEPS,epsilon,'ObjectTextSizeBytes',False))

        print('sample start')
        async_result = pool.starmap_async(runner,[arg for arg in args])


        # BUG: pool.starmap will freqeuntly hang/deadlock after everything in the runner is finished
        async_result.wait(180)
        if not async_result.ready():
            pool.terminate()
            print('terminated')
            pool.join()
            pool = Pool(processes = cpu_count() + 20)

        else:
            print('sample end')

        reduction = validate(test_corpus)/test_o0_absolute_size
        print(f'{reduction.mean()}/{reduction.std()}')
        steps += (EP_MAX_STEPS)*EPISODE_PER_EVAL
        counter += EPISODE_PER_EVAL
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    set_start_method("spawn")

    #env = compiler_gym.make("llvm-v0",benchmark="cbench-v1/qsort")
    env = compiler_gym.make("llvm-v0",observation_space="IrSha1",reward_space="ObjectTextSizeNorm")
    env.reset()

    #corpus = get_dataset(env,"benchmark://blas-v0")
    corpus = get_dataset(env)
    train_corpus, test_corpus = split_train_and_test(corpus)
    print(f'Corpus Size : {len(corpus)} ({len(train_corpus)}/{len(test_corpus)})')

    env.close()

    test_o0_absolute_size = get_base_absolute_size(test_corpus,'O0')
    test_oz_size_reduction = get_oz_size_reduction(test_corpus)
    print(f'Oz : {test_oz_size_reduction.mean()}/{test_oz_size_reduction.std()}')
    

    BATCH_SIZE = 1
    #GAMMA = 0.999
    GAMMA = 0.5
    E_START = 0.99
    E_END = 0.5
    #E_DECAY = 2000
    E_DECAY = 200000
    #TARGET_UPDATE = 10000
    TARGET_UPDATE = 2000
    #TARGET_UPDATE = 500
    EP_MAX_STEPS = 100
    #EP_MAX_STEPS = 20
    LEARNING_RATE = 0.0005



    steps = 0


    reduction = validate(test_corpus)/test_o0_absolute_size 
    print(f'Init : {reduction.mean()}/{reduction.std()}')

    TOTAL_EP_TO_TRAIN = 100000
    main(TOTAL_EP_TO_TRAIN)

    #updates = 0
    #while updates < 1000000:
    #    sample_traj(100)
    #    print('done sampling')
    #    train()


