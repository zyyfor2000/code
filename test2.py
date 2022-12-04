#%%
from multiprocessing import Pool # 导入多进程中的进程
from multiprocessing import cpu_count
import numpy as np
n_total = (5,10,20,100,200,300)

import myfunction
#%%
if __name__ == '__main__':
    p = Pool(processes=8)
    result = p.map_async(myfunction.tc_cal,n_total)
    print("done")
    p.close()
    p.join()
    print("Sub-process(es) done.")
    print(result._value)

#%% 试试别的并行
if __name__ == '__main__':
    r = list(range(100))
    p = Pool(processes=cpu_count()-1)
    result = p.map_async(myfunction.tc_cal2,r)
    print("done")
    p.close()
    p.join()
    print("Sub-process(es) done.")
    print(np.mean(result._value,axis=0))