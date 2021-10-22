import random

class replay_buffer:
    def __init__(self,capacity=50000):
        self.memory = []
        self.getting = False

    def get_batch(self,batch_size=128):
        if not self.getting:
            random.shuffle(self.memory)
            self.getting = True

        if len(self.memory) >= batch_size:
            r = self.memory[:batch_size]
            self.memory = self.memory[batch_size:]
        else:
            r = self.memory
            self.memory = []
        return r

    def insert(self,step):
        self.getting = False
        self.memory.append(step)

    def __len__(self):
        return len(self.memory)

