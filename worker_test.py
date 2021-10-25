from multiprocessing.connection import Client
import subprocess
import time

ports = [2650,2651,2652,2653]

for port in ports:
    subprocess.Popen(['python3','worker.py',str(port)])

for port in ports:
    while True:
        try:
            conn = Client(('localhost',port), authkey=b'secret password')
            print('worker is spawned!')
            break
        except Exception as e:
            pass


    conn.send('hi')
    print(conn.recv())
    conn.close()

conn = Client(('localhost',ports[0]), authkey=b'secret password')
conn.send(('sample','cbench-v1/qsort',20,1,'ObjectTextSizeNorm',False))
start = time.time()
while True:
    if conn.poll():
        print(conn.recv())
        end = time.time()
        print(end-start)
        break


for port in ports:
    conn = Client(('localhost',port), authkey=b'secret password')
    conn.send(('exit',None))
    conn.close()

