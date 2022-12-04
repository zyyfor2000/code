# %%初始参数设定
from scipy.stats import norm, chi2, truncnorm
from math import sqrt
import numpy as np
D_n = 10
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000

#%% 经典方法求解，tc恒为定值
for n_change in [5, 10, 20, 100]:
    S_classic = L * D_n + sqrt(L*(L+1)*(2*L+1)/6) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    z_classic = L*D_n
    for i in range(L):
        z_classic = z_classic + (L-i)*norm.rvs(0, sqrt(sigma2_hat), size=n_replication)
    z_classic_new = S_classic - z_classic
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                  sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic,  TC_classic, "n_change = {}".format(n_change))
#%% 经典方法改进 s不变
n_replication_c =1000
for n_change in [5, 10, 20, 100]:
    tc = []
    S_classic = L * D_n + sqrt(L*(L+1)*(2*L+1)/6) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    data_in_practice = np.zeros((n_replication_c,n_change))
    data_in_practice[:,n_change-1] = D_n
    for i in range(n_change-1):
        data_in_practice[:,n_change-i-2]= data_in_practice[:,n_change-i-1]-norm.rvs(0, sqrt(sigma2_hat), size=n_replication_c)
    for i in range(n_replication_c):
        sigma2_hat_hat = np.var(data_in_practice[i],ddof=1)
        z_classic = norm.rvs(L * D_n , sqrt(L*(L+1)*(2*L+1)/6*sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
    print(S_classic,  np.mean(tc), "n_change = {}".format(n_change))
#%% 经典方法改进 s变
n_replication_c = 1000
sigma2_hat_hat = sigma2_hat
for n_change in [5, 10, 20, 100]:
    tc = []
    s_total = []
    data_in_practice = np.zeros((n_replication_c, n_change))
    data_in_practice[:, n_change - 1] = D_n
    for i in range(n_change - 1):
        data_in_practice[:, n_change-i-2] = data_in_practice[:, n_change-i-1] - norm.rvs(0, sqrt(sigma2_hat),size=n_replication_c)
    for i in range(n_replication_c):
        sigma2_hat_hat = np.var(data_in_practice[i], ddof=1)
        S_classic = L * D_n + sqrt(L * (L + 1) * (2 * L + 1) / 6) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat_hat)
        z_classic = norm.rvs(L * D_n, sqrt(L * (L + 1) * (2 * L + 1) / 6 * sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
        s_total.append(S_classic)
    print(np.mean(s_total), np.mean(tc), "n_change = {}".format(n_change))

# %% 精确方法求解
for n in [5, 10, 20, 100]:
    z_exact = []
    sigma2_hat_1 = (n - 2) * sigma2_hat / chi2.rvs(n - 2, size=n_replication)
    for i in range(n_replication):
        z_exact.append(norm.rvs(loc=L * D_n, scale=sqrt(L * (L + 1) * (2 * L + 1) / 6 * sigma2_hat_1[i]), size=1))
    z_exact = np.array(z_exact)
    S_exact = np.quantile(z_exact, b / (b + h))
    safety_exact = S_exact - np.mean(z_exact)
    z_exact_new = S_exact - z_exact
    TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                sum(b * z_exact_new[z_exact_new < 0])) / n_replication
    print(S_exact, safety_exact, TC_exact, "n_change = {}".format(n),"exact stop")

# %%渐进方法求解
b = 20
for n in [5, 10, 20, 100]:
    z_asymtotic = []
    left_bound = -sqrt(n / 2)
    #left_bound = 0
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
    for i in range(n_replication):
        z_asymtotic.append(norm.rvs(loc=L * D_n,
                                    scale=sqrt(L * (L + 1) * (2 * L + 1) / 6 * sigma2_hat_1[i]), size=1))
    z_asymtotic = np.array(z_asymtotic)
    S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
    safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
    z_asymtotic_new = S_asymtotic - z_asymtotic
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    print(S_asymtotic, safety_asymtotic, TC_asymtotic,"n_change = {}".format(n), "asymtotic stop")
