import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self,input_width,num_actions):
        super(DQN,self).__init__()
        self.dense1 = nn.Linear(input_width,100)
        self.dense2 = nn.Linear(100,100)
        self.dense3 = nn.Linear(100,num_actions)

    def forward(self,x):
        x = x.to(device)
        #x = torch.tanh(self.dense1(x))
        #x = torch.tanh(self.dense2(x))
        #x = torch.tanh(self.dense3(x)) # expect negative output, so no activation??
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class DQN_LSTM(nn.Module):
    def __init__(self,vector,width,num_actions):
        super(DQN_LSTM,self).__init__()

        LSTM_INIT = {
                'input_size':vector_width,
                'hidden_size':vector_width,
                'batch_first':True,
                'num_layers':1,
                'bidirectional':False,
                'dropout':0
        }

        self.lstm = nn.LSTM(**LSTM_INIT)
        self.dense1 = nn.Linear(LSTM_INIT["hidden_size"],300)
        self.dense2 = nn.Linear(300,num_actions)

    def forward(self,x):
        x = x.to(device)
        x = F.relu(self.lstm(x)[1][0].squeeze()) # output,(h_n,c_n)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


