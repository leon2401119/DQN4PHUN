import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import compiler_gym
import pickle
from multiprocessing.connection import Listener
from multiprocessing import shared_memory
from DQN import DQN
import random
import os
#from multiprocessing.managers import BaseManager
#from queue import Queue
#queue = Queue()
#class QueueManager(BaseManager): pass
#QueueManager.register('get_queue', callable=lambda:queue)

#def train(data,batch_size,gamma,target_update):
def train(batch_size,gamma,target_update):
    global steps

    #data = pickle.loads(data)
    #print(data)
    #random.shuffle(data)

    #data = shm.buf
    #q = manager.get_queue()

    global replay_mem
    #data = replay_mem
    while len(replay_mem):
        if len(replay_mem)>=batch_size:
            batch = replay_mem[:batch_size]
            replay_mem = replay_mem[batch_size:]

        else:
            batch = replay_mem
            replay_mem = []

    #while not q.empty():
    #    batch = []

    #    for _ in range(batch_size):
    #        try:
    #            batch.append(q.get())
    #        except Exception as e:
    #            print(e)

        s0, s1, r, a, done = [],[],[],[],[]
        for i in range(batch_size):
            s0.append(batch[i][0])
            s1.append(batch[i][4])
            done.append(batch[i][3])
            r.append([batch[i][2]])
            a.append([batch[i][1]])

        s0 = torch.tensor(s0,dtype=torch.float32).to(device)
        s1 = torch.tensor(s1,dtype=torch.float32).to(device)
        r = torch.tensor(r).to(device)
        a = torch.tensor(a).to(device)

        Q_s0 = torch.gather(policy_net(s0),1,a)
        Q_s1 = target_net(s1).max(dim=1,keepdim=True)[0].detach()

        Q_target = Q_s1 + gamma*r
        for j,mask in enumerate(done):
            if mask:
                Q_target[j] = r[j]

        criterion = nn.MSELoss()
        loss = criterion(Q_s0, Q_target)
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        steps += 1

        os.system(f'echo "{loss}" >>loss_history')

        if not steps % target_update:
            target_net.load_state_dict(policy_net.state_dict())

        if not steps % 1000:
            print(f'{loss.item()}')

    return loss.item()


env = compiler_gym.make("llvm-v0")
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(len(env.observation["Autophase"]),len(env.action_space.names)).to(device)
target_net = DQN(len(env.observation["Autophase"]),len(env.action_space.names)).to(device)
target_net.eval()

LEARNING_RATE = 0.005
optimizer = optim.Adam(policy_net.parameters(),lr=LEARNING_RATE)
env.close()

steps = 0
os.system(f'rm -f loss_history')

#shm = shared_memory.SharedMemory(create=True, size=1e8,name='replay_memory')
#manager = QueueManager(address=('', 50000), authkey=b'abc')
#manager.start()
#server = manager.get_server()
#server.serve_forever()

replay_mem = []

listener = Listener(('localhost', 6000), authkey=b'secret password')
running = True

while running:
    conn = listener.accept()
    #print('connection accepted from',listener.last_accepted)
    msg = conn.recv()

    action = msg[0]

    if action == 'inference':
        q = policy_net(torch.tensor(msg[1],dtype=torch.float32))
        conn.send(q.detach().cpu().numpy())
        conn.close()

    elif action == 'enqueue':
        replay_mem.extend(msg[1])
        #print('recved')

    elif action == 'train':
        loss = train(*msg[1:])
        conn.send((steps,loss))
        conn.close()

    if len(replay_mem) >= 500:
        train(1,0.5,5000)


