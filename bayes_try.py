# %%初始参数设定
import math
from sympy.stats import GammaInverse
from scipy.stats import norm, chi2, truncnorm, binom, invgamma,gamma
from math import sqrt
import numpy as np
import pandas as pd

n_replication = 1000000

# %%  对D没有重复
mu_hat = 20
sigma2_hat = 4
b = 20
h = 1
L = 5
# 真实数据生成
z_real = []
mu = np.array(norm.rvs(size = n_replication, loc=0, scale = 100000))
sigma2 = 1/np.array(gamma.rvs(size = n_replication, a = 0.0001000, scale = 1000))
for i in range(n_replication):
    z_real.append(norm.rvs(size = 1, loc = L*mu[i], scale = sqrt(L* sigma2[i])))
#%%
for n in [5, 10, 20, 100]:
    print("n={}".format(n))
    # 精确方法求解
    z_exact = []
    sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
    for i in range(n_replication):
        mu_hat_1 = mu_hat + sqrt(sigma2_hat_1[i]) * norm.rvs(size=1) / sqrt(n)
        z_exact.append(norm.rvs(loc=L * mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_exact = np.array(z_exact)
    S_exact = np.quantile(z_exact, b / (b + h))
    safety_exact = S_exact - np.mean(z_exact)
    z_exact_new = S_exact - z_real
    TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                sum(b * z_exact_new[z_exact_new < 0])) / n_replication
    print(S_exact, TC_exact, "exact stop")

    #  经典方法求解
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
    z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
    safety_classic = S_classic - np.mean(z_classic)
    z_classic_new = S_classic - z_real
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                  sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic, TC_classic, "classic stop")

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
    z_asymtotic_new = S_asymtotic - z_real
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    print(S_asymtotic, TC_asymtotic, "asymtotic stop\n")
