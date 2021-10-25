import compiler_gym
import numpy as np
import math
import random
import time
from multiprocessing import cpu_count
import os
from multiprocessing.connection import Client
import subprocess

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


def validate(corpus, worker_ports):
    VAL_MAX_STEPS = 5
    conns = [None for _ in worker_ports]
    
    sizes = []

    msg = ('sample',train_corpus[random.randint(0,len(train_corpus)-1)],VAL_MAX_STEPS,0,'ObjectTextSizeBytes',True)

    for benchmark in corpus:
        poll_counter = 0
        while True:
            if conns[poll_counter] is None:
                conns[poll_counter] = Client(('localhost',worker_ports[poll_counter]), authkey=b'secret password')
                conns[poll_counter].send(msg)
                break
            
            elif conns[poll_counter].poll():
                sizes.append(conns[poll_counter].recv())
                conns[poll_counter].close()
                conns[poll_counter] = None

            poll_counter = (poll_counter+1)%len(conns)

            
    for conn in conns:
        if conn is not None:
            sizes.append(conn.recv())
            conn.close()

    sizes = np.array(sizes)
    return sizes


def train():
    conn = Client(('localhost', 6000), authkey=b'secret password')
    #conn.send(('train',pickle.dumps(replay_memory),BATCH_SIZE,GAMMA,TARGET_UPDATE))
    conn.send(('train',BATCH_SIZE,GAMMA,TARGET_UPDATE))
    _,_ = conn.recv()

    reduction = validate(test_corpus)/test_o0_absolute_size
    print(f'{reduction.mean()}/{reduction.std()}')


def sample(ep, worker_ports):
    global steps

    EPISODE_PER_EVAL = 100
    counter = 0
    conns = [None for _ in worker_ports]

    print('sample start')

    while counter < ep:
        epsilon = E_END + (E_START - E_END) * math.exp(-1. * steps / E_DECAY)
        msg = ('sample',train_corpus[random.randint(0,len(train_corpus)-1)],EP_MAX_STEPS,epsilon,'ObjectTextSizeBytes',False)

        poll_counter = 0
        while True:
            if conns[poll_counter] is None:
                conns[poll_counter] = Client(('localhost',worker_ports[poll_counter]), authkey=b'secret password')
                conns[poll_counter].send(msg)
                print(f'task scheduled to {poll_counter}')
                break

            elif conns[poll_counter].poll():
                steps += conns[poll_counter].recv()
                conns[poll_counter].close()
                conns[poll_counter] = None

            poll_counter = (poll_counter+1)%len(conns)
            
        counter += 1

    # finish the job
    for conn in conns:
        if conn is not None:
            print('cleaning up')
            steps += conn.recv()
            conn.close()


    return

        #if not counter%EPISIDE_PER_EVAL:
        #    print('sample end')

        #    reduction = validate(test_corpus)/test_o0_absolute_size
        #    print(f'{reduction.mean()}/{reduction.std()}')

        #    print('sample start')



def start_workers(ports):
    def handshake(port):
        while True:
            try:
                conn = Client(('localhost',port), authkey=b'secret password')
                break
            except Exception as e:
                pass

        conn.send('hi')
        msg = conn.recv()
        if msg == 'hi back':
            print(f'worker on port {port} ready')
        conn.close()

    for port in ports:
        subprocess.Popen(['python3','worker.py',str(port)])

    for port in ports:
        handshake(port)


def kill_workers(ports):
    for port in ports:
        conn = Client(('localhost',port), authkey=b'secret password')
        conn.send(('exit',))



if __name__ == '__main__':

    env = compiler_gym.make("llvm-v0",observation_space="IrSha1",reward_space="ObjectTextSizeNorm")
    env.reset()

    corpus = get_dataset(env,"benchmark://blas-v0")
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
    #EP_MAX_STEPS = 100
    EP_MAX_STEPS = 10
    LEARNING_RATE = 0.0005

   
    PORT_START = 2450
    WORKER_COUNT = cpu_count() + 10
    #WORKER_COUNT = 5
    ports = [port for port in range(PORT_START,PORT_START + WORKER_COUNT)]
    start_workers(ports)

    steps = 0


    #reduction = validate(test_corpus)/test_o0_absolute_size 
    #print(f'Init : {reduction.mean()}/{reduction.std()}')

    while True:
        sample(100,ports)
    reduction = validate(test_corpus,ports)/test_o0_absolute_size
    print(f'{reduction.mean()}/{reduction.std()}')

    kill_workers(ports)

