
#%%
import threading
import time
from queue import Queue

def thread_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")
    #print("This is an added Thread,number is %s"%threading.current_thread())

def T2_job():
    print("T2 start\n")
    print("T2 finish")

def job(l,q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l) #计算过的列表放到q里

def multithreading():
    q = Queue()
    threads = []
    data = [[1,2,3],[4,5,1],[1,2],[2,4,5]]
    for i in range(4):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join() #加载到主线程里
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)




def main():
    added_thread = threading.Thread(target=thread_job, name="T1")
    thrad2 = threading.Thread(target=T2_job,name="T2")
    added_thread.start()
    thrad2.start()
    added_thread.join()
    thrad2.join()

    print("all done\n")
    # print(threading.active_count()) #看现在有多少激活的线程
    # print(threading.enumerate()) #看是哪几个
    # print(threading.current_thread())

# if __name__ == "__main__":
#     #main()
#     multithreading()

#%% 多核

import multiprocessing as mp
import myfunction


if __name__ == "__main__":
    q = mp.Queue()

    p1 = mp.Process(target=myfunction.job,args=(q,))
    p1.start()
    p1.join()
    res1 = q.get()

#%% 效率对比
import myfunction


import multiprocessing as mp



def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=myfunction.job, args=(q,))
    p2 = mp.Process(target=myfunction.job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multicore:',res1 + res2)
import threading as td

def multithread():
    q = mp.Queue() # thread可放入process同样的queue中
    t1 = td.Thread(target=myfunction.job, args=(q,))
    t2 = td.Thread(target=myfunction.job, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithread:', res1 + res2)

def normal():
    res = 0
    for _ in range(2):
        for i in range(10000000):
            res += i + i**2 + i**3
    print('normal:', res)
import time

if __name__ == '__main__':
    st = time.time()
    normal()
    st1 = time.time()
    print('normal time:', st1 - st)
    multithread()
    st2 = time.time()
    print('multithread time:', st2 - st1)
    multicore()
    print('multicore time:', time.time() - st2)

#%% pool 用法
import multiprocessing as mp
import myfunction
import time
def multicore():
    test = 100000000
    begin = time.time()
    pool = mp.Pool()
    res1= pool.map(myfunction.job_pool, range(test))
    end = time.time()
    print(end - begin)
    print(res1[0:5])
    begin = time.time()
    res2 = []
    for i in range(test):
        res2.append(i*i)
    end = time.time()
    print(end - begin)
    print(res2[0:5])

#%%
from multiprocessing import cpu_count

print("CPU的核数为：{}".format(cpu_count()))
print(type(cpu_count()))

if __name__ == '__main__':
    multicore() # try to modify