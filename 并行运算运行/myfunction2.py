#%% 加非参数版
from numpy import *
import random
from scipy.stats import norm, chi2, truncnorm, binom
from math import sqrt
import numpy as np

n_replication = 100 #1000000

random.seed(100)
n_population = n_replication
mu = 10
sigma2 = 4
b = 50
h = 1
L = 5
x_all = norm.rvs(size=n_population, loc=10, scale=sqrt(sigma2))
x_l_all = [0] * n_population
i = 0
for i in range(n_population):
    x_l_all[i] = sum(np.random.choice(x_all, 5))


def tc_cal2(r):
    r_all = 1
    tc_all = np.zeros((r_all, 3))
    tc_np = np.zeros((r_all, 6))
    print(n)
    show = 0

    data_known = np.random.choice(x_all, n)
    mu_hat = np.mean(data_known)
    sigma2_hat = np.var(data_known, ddof=1)
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
    tc_all[0, 0] = TC_exact

    #  经典方法求解
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
    z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
    safety_classic = S_classic - np.mean(z_classic)
    z_classic_new = S_classic - x_l_all
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                  sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    tc_all[0, 1] = TC_classic
    # 渐进方法求解
    z_asymtotic = []
    left_bound = -sqrt(n / 2)
    # left_bound = 0
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), loc=0,
                                                                              scale=1,
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
    tc_all[0, 2] = TC_asymtotic

    S_1 = np.quantile(data_known, q=b / (b + h)) * L
    z_1 = S_1 - x_l_all
    TC_1 = (sum(h * z_1[z_1 > 0]) -
            sum(b * z_1[z_1 < 0])) / n_replication
    tc_np[0, 0] = TC_1

    n_1, n_2 = int(np.floor(n * 0.5)), int(np.floor(n * 0.8))
    S = []
    for _ in range(100):
        S.append(np.quantile(np.random.choice(data_known, n_1, replace=True), q=b / (b + h)))
    S_21 = np.mean(S) * L
    z_21 = S_21 - x_l_all
    TC_21 = (sum(h * z_21[z_21 > 0]) -
             sum(b * z_21[z_21 < 0])) / n_replication
    tc_np[0, 1] = TC_21

    S = []
    for _ in range(100):
        S.append(np.quantile(np.random.choice(data_known, n_2, replace=True), q=b / (b + h)))
    S_22 = np.mean(S) * L
    z_22 = S_22 - x_l_all
    TC_22 = (sum(h * z_22[z_22 > 0]) -
             sum(b * z_22[z_22 < 0])) / n_replication
    tc_np[0, 2] = TC_22
    # 3. SVP1-3
    data_ordered = np.sort(data_known)
    s_1 = 0
    p = b / (b + h)
    for i in range(1, n - 1):
        s_1 += (binom.pmf(i + 1, n, p) + binom.pmf(i, n, p)) / 2 * data_ordered[i]
    S_31 = (2 * binom.pmf(0, n, p) + binom.pmf(1, n, p)) / 2 * data_ordered[0] + binom.pmf(0, n, p) \
           / 2 * data_ordered[1] - binom.pmf(0, n, p) / 2 * data_ordered[2] + s_1 - binom.pmf(n, n, p) / 2 \
           * data_ordered[n - 3] + binom.pmf(n, n, p) / 2 * data_ordered[n - 2] + (2 * binom.pmf(n, n, p) \
                                                                                   + binom.pmf(n - 1, n, p)) / 2 * \
           data_ordered[n - 1]
    z_31 = S_31 * L - x_l_all
    TC_31 = (sum(h * z_31[z_31 > 0]) - sum(b * z_31[z_31 < 0])) / n_replication
    tc_np[0, 3] = TC_31
    s_2 = 0
    for i in range(n):
        s_2 = s_2 + binom.pmf(i, n, p) * data_ordered[i]
    S_32 = s_2 + (2 * data_ordered[n - 1] - data_ordered[n - 2]) * binom.pmf(n, n, p)
    z_32 = S_32 * L - x_l_all
    TC_32 = (sum(h * z_32[z_32 > 0]) - sum(b * z_32[z_32 < 0])) / n_replication
    tc_np[0, 4] = TC_32
    s_3 = 0
    for i in range(n):
        s_3 = s_3 + binom.pmf(i + 1, n, p) * data_ordered[i]
    S_33 = s_3 + (2 * data_ordered[0] - data_ordered[1]) * binom.pmf(0, n, p)
    z_33 = S_33 * L - x_l_all
    TC_33 = (sum(h * z_33[z_33 > 0]) - sum(b * z_33[z_33 < 0])) / n_replication
    tc_np[0, 5] = TC_33

    return np.append(tc_all, tc_np)