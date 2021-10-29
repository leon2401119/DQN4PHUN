from multiprocessing.connection import Client
import compiler_gym

env = compiler_gym.make('llvm-v0',reward_space='ObjectTextSizeNorm')
env.reset(benchmark='cbench-v1/qsort')

for _ in range(2):
    conn = Client(('localhost', 6000), authkey=b'secret password')
    conn.send(('inference',env.observation['Autophase']))
    q = conn.recv()
    r = env.step(q.argmax().item())[1]
    #r = env.step(env.action_space.sample())
    #print(r)
    conn.close()
    print(q)

env.close()
