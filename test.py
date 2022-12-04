from numpy import *
from multiprocessing import Pool # 导入多进程中的进程池
from multiprocessing.pool import ThreadPool as Pool
import math
import random
from scipy.stats import norm, chi2, truncnorm, binom
from math import sqrt
import numpy as np
import pandas as pd

n_replication = 1000000

random.seed(100)
#%% 原始数据产生
n_population = n_replication
mu = 10
sigma2 =4
b = 50
h = 1
L = 5
x_all = norm.rvs(size=n_population, loc=10, scale=sqrt(sigma2))
x_l_all = [0] * n_population
i = 0
for i in range(n_population):
    x_l_all[i] = sum(np.random.choice(x_all,5))


# %%
def tc_cal(n):
    r_all = 100
    tc_all = np.zeros((r_all,3))
    print(n)
    for r in range(r_all):
        data_known = np.random.choice(x_all, n)
        mu_hat = np.mean(data_known)
        sigma2_hat = np.var(data_known, ddof=1)
        # 精确方法求解
        z_exact = []
        sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
        for i in range(n_replication):
            mu_hat_1 = mu_hat + sqrt(sigma2_hat_1[i]) * norm.rvs(size=1) / sqrt(n)
            z_exact.append(norm.rvs(loc=L * mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
        z_exact = np.array(z_exact)
        S_exact = np.quantile(z_exact, b / (b + h))
        safety_exact = S_exact - np.mean(z_exact)
        z_exact_new = S_exact - x_l_all
        TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                    sum(b * z_exact_new[z_exact_new < 0])) / n_replication
        tc_all[r,0] = TC_exact

        #  经典方法求解
        S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
        eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
        z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
        safety_classic = S_classic - np.mean(z_classic)
        z_classic_new = S_classic - x_l_all
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc_all[r, 1] = TC_classic
        # 渐进方法求解
        z_asymtotic = []
        left_bound = -sqrt(n / 2)
        # left_bound = 0
        sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), loc=0, scale=1,
                                                                                  size=n_replication)
        mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
        for i in range(n_replication):
            z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                        scale=sqrt(L * sigma2_hat_1[i]), size=1))
        z_asymtotic = np.array(z_asymtotic)
        S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
        safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
        z_asymtotic_new = S_asymtotic - x_l_all
        TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                        sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
        tc_all[r, 2] = TC_asymtotic
    print("yes")
    return np.mean(tc_all,axis=0)


#%%


n_total = (5,10,20,100)


if __name__ == '__main__':
    p = Pool(4000)
    result = p.map_async(tc_cal,n_total)
    print("done")
    p.close()
    p.join()
    print("Sub-process(es) done.")
    print(result._value)


