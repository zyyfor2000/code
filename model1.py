# %%初始参数设定
import math

from scipy.stats import norm, chi2, truncnorm
from math import sqrt
import numpy as np

b = 20
h = 1
L = 5
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
n_replication = 1000000

# %% 经典方法求解
S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
# s_classic_other_way = eq1.isf(1 - b / (b + h))
z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
safety_classic = S_classic - np.mean(z_classic)
z_classic_new = S_classic - z_classic
TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
              sum(b * z_classic_new[z_classic_new < 0])) / n_replication
print(S_classic, safety_classic, TC_classic, "classic stop")
#%% 经典方法改进 s不变
for n_change in [5, 10, 20, 100]:
    tc = []
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    for i in range(10000):
        data_in_practice = norm.rvs(mu_hat, sqrt(sigma2_hat), size=n_change)
        mu_hat_hat = np.mean(data_in_practice)
        sigma2_hat_hat = np.var(data_in_practice,ddof=1)
        z_classic = norm.rvs(L*mu_hat_hat, sqrt(L*sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
    print(S_classic,  np.mean(tc), "n_change = {}".format(n_change))
#%% 经典方法改进 s变
for n_change in [5, 10, 20, 100]:
    tc = []
    s_total = []
    mu_hat_hat = mu_hat
    sigma2_hat_hat = sigma2_hat
    for i in range(1000):    #1000， 10000， 1000000
        S_classic = L * mu_hat_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat_hat)
        data_in_practice = norm.rvs(mu_hat, sqrt(sigma2_hat), size=n_change)
        mu_hat_hat = np.mean(data_in_practice)
        sigma2_hat_hat = np.var(data_in_practice,ddof=1)
        z_classic = norm.rvs(L*mu_hat_hat, sqrt(L*sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
        s_total.append(S_classic)
    print(np.mean(s_total),  np.mean(tc), "n_change = {}".format(n_change))
# %% 精确方法求解
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
z_exact = []
sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
for i in range(n_replication):
    mu_hat_1 = mu_hat + sqrt(sigma2_hat_1[i]) * norm.rvs(size=1) / sqrt(n)
    z_exact.append(norm.rvs(loc=L * mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_exact = np.array(z_exact)
S_exact = np.quantile(z_exact, b / (b + h))
safety_exact = S_exact - np.mean(z_exact)
z_exact_new = S_exact - z_exact
TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
            sum(b * z_exact_new[z_exact_new < 0])) / n_replication
print(S_exact, safety_exact, TC_exact, "exact stop")

# %%渐进方法求解

mu_hat = 10
sigma2_hat = 4
n = 10  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
z_asymtotic = []
left_bound = -sqrt(n / 3)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
for i in range(n_replication):
    z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_asymtotic = np.array(z_asymtotic)
S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
z_asymtotic_new = S_asymtotic - z_asymtotic
TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
print(S_asymtotic, safety_asymtotic, TC_asymtotic, "asymtotic stop")

# %%渐进方法求解 n次数小时增加循环次数
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
tc = []
for j in range(1000):
    z_asymtotic = []
    left_bound = -sqrt(n / 2)
    #left_bound = 0
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
    mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
    for i in range(n_replication):
        z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                    scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_asymtotic = np.array(z_asymtotic)
    S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
    safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
    z_asymtotic_new = S_asymtotic - z_asymtotic
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    tc.append(TC_asymtotic)
print(S_asymtotic, safety_asymtotic, np.mean(tc), "asymtotic stop")
# %%渐进方法改进1 mu和sigma有联系
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
z_asymtotic = []
left_bound = -sqrt(n / 2)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
for i in range(n_replication):
    mu_hat_1 = mu_hat + sqrt(sigma2_hat_1[i] / n) * norm.rvs(size=1)
    z_asymtotic.append(norm.rvs(loc=L * mu_hat_1,
                                scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_asymtotic = np.array(z_asymtotic)
S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
z_asymtotic_new = S_asymtotic - z_asymtotic
TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
print(S_asymtotic, safety_asymtotic, TC_asymtotic, "asymtotic stop")



# %%渐进方法求解2 重新生成实际数据1
tc = []
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
z_asymtotic = []
left_bound = -sqrt(n / 2)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
for i in range(n_replication):
    z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_asymtotic = np.array(z_asymtotic)
S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
for i in range(n_replication):
    z_asymtotic = norm.rvs(L*mu_hat_1[i], sqrt(L*sigma2_hat_1[i]), size = n)
    z_asymtotic_new = S_asymtotic - z_asymtotic
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n
    tc.append(TC_asymtotic)
print(S_asymtotic,np.mean(tc), "asymtotic stop")
# %%渐进方法求解2 重新生成实际数据2
mu_hat = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
z_asymtotic = []
left_bound = -sqrt(3*n / 2)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
for i in range(n_replication):
    z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_asymtotic = np.array(z_asymtotic)
S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
z_asymtotic = norm.rvs(L*mu_hat, sqrt(L*sigma2_hat), size = n_replication)
z_asymtotic_new = S_asymtotic - z_asymtotic
TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
print(S_asymtotic,TC_asymtotic, "asymtotic stop")


# %%效果比较
# 库存效果
safety_approx_add = (safety_asymtotic - safety_classic) / safety_classic
safety_exact_add = (safety_exact - safety_classic) / safety_classic
print(safety_approx_add, safety_exact_add)
# 花费效果
TC_asymtotic_sub = (TC_asymtotic - TC_classic) / TC_classic
TC_exact_sub = (TC_exact - TC_classic) / TC_classic
print(TC_asymtotic_sub, TC_exact_sub)

#%%
from math import factorial
m = 50
n = 50
ans1 = 0
for k in range(m,m+n+1):
    ans1 += factorial(k)/factorial(k-m)
ans1 = m+n - m*factorial(n)/factorial(m+n)*ans1
ans2 = n/(m+1)
print(ans1, ans2)

#%%
def f(k):
    s = 0
    for i in range(k):
        s += (k-i)*10**i/factorial(10)
    return s
(f(20)+f(30))/11**50

#%%
from scipy.special import comb
def num(k):
    s = 0
    for i in range(k):
        s += comb(k+8-i,9)
    return s
print(num(20)/comb(30,10), num(30)/comb(40,10))
