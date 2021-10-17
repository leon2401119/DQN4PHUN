import compiler_gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from collections import deque


# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self,vector_width, num_actions):
        super(DQN,self).__init__()
        
        LSTM_INIT = {
            'input_size' : vector_width,
            'hidden_size' : vector_width, # = final embedding size. set to larger? smaller?
            'batch_first' : True,
            'num_layers' : 1,
            'bidirectional' : False,
            'dropout' : 0
        }
        self.lstm = nn.LSTM(**LSTM_INIT)

        self.dense1 = nn.Linear(LSTM_INIT["hidden_size"], 300)
        self.dense2 = nn.Linear(300, num_actions)

    def forward(self,x):
        x = x.to(device)
        x = F.relu(self.lstm(x)[1][0].squeeze()) # lstm output is   output,(h_n,c_n) 
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


def get_dataset(env,dataset="benchmark://cbench-v1"):
    uris = []
    for uri in env.datasets[dataset].benchmark_uris():
        uris.append(uri)
    return uris



class replay_buffer:
    def __init__(self, capacity=50000):
        #self.hash_table = {}
        self.memory = deque([],maxlen=capacity)

    def get_batch(self,batch_size=128):
        return random.sample(self.memory, batch_size)

    def insert(self,step):
        self.memory.append(step)

    def __len__(self):
        return len(self.memory)


#env = compiler_gym.make("llvm-v0",benchmark="cbench-v1/qsort")
env = compiler_gym.make("llvm-v0",observation_space="IrSha1",reward_space="ObjectTextSizeOz")
env.reset()

corpus = get_dataset(env)

BATCH_SIZE = 128
GAMMA = 0.999
#E_START = 0.9
E_START = 1
E_END = 0.05
E_DECAY = 200
TARGET_UPDATE = 10
EP_MAX_STEPS = 30

policy_net = DQN(len(env.observation["Inst2vec"][0]),len(env.action_space.names)).to(device)
target_net = DQN(len(env.observation["Inst2vec"][0]),len(env.action_space.names)).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = replay_buffer(100)

steps = 0

for _ in range(5):
    # TODO : this is NOT a scalable approach
    #env.reset(benchmark=corpus[random.randint(0,len(corpus)-1)])
    env.reset(benchmark="cbench-v1/patricia")
    print(env.benchmark)
    for _ in range(EP_MAX_STEPS):

        #observation = env.observation["IrSha1"]
        #observation_i2v = tf.expand_dims(tf.convert_to_tensor(env.observation["Inst2vec"]),0)
        #observation_i2v = tf.convert_to_tensor(env.observation["Inst2vec"])
        #replay_buf.hash_table[observation] = observation_i2v
        observation = env.observation["Inst2vec"]

        if random.random() < E_START: # TODO: decay
            action = env.action_space.sample()
        else:
            action = policy_net(torch.tensor([observation])).argmax().item()
            #print(type(action.)) # action is int

        _, reward, done, _ = env.step(action)
        observation_next = env.observation["Inst2vec"]

        memory.insert([observation,action,reward,observation_next])

    #observation_last = tf.expand_dims(tf.convert_to_tensor(env.observation["Inst2vec"]),0)
    #observation_last = tf.convert_to_tensor(env.observation["Inst2vec"])
    #replay_buf.hash_table[env.observation["IrSha1"]] = observation_last


BATCH_SIZE = 16
batch = memory.get_batch(BATCH_SIZE)
s0, s1, r, a = [],[],[],[]
for i in range(BATCH_SIZE):
    s0.append(torch.tensor(batch[i][0]))
    s1.append(torch.tensor(batch[i][3]))
    r.append([batch[i][2]])
    a.append([batch[i][1]])

s0 = pad_sequence(s0,batch_first=True) # s0.shape = (batch,1489+-,200)
s1 = pad_sequence(s1,batch_first=True)
r = torch.tensor(r).to(device)
a = torch.tensor(a).to(device)

# policy_net(s0) has output shape of (batch,num_actions)
Q_s0 = torch.gather(policy_net(s0),1,a)
Q_s1 = target_net(s1).max(dim=1,keepdim=True)[0].detach() # max returns (max,argmax)

#print(Q_s0)
#print(Q_s1)

criterion = nn.SmoothL1Loss()
loss = criterion(Q_s0, Q_s1 + GAMMA*r)

optimizer.zero_grad()
loss.backward()
for param in policy_net.parameters():
    param.grad.data.clamp_(-1, 1)
optimizer.step()



# for i in env.datasets["benchmark://cbench-v1"].benchmarks():
#   print(i)
# for i in env.datasets["benchmark://cbench-v1"].benchmark_uris():


env.close()
