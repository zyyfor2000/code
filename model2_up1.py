# %%初始参数设定
from scipy.stats import norm, chi2, truncnorm, binom
from math import sqrt
import numpy as np
mu_hat = 10
sigma2_hat = 4
alpha = 0.8
b = 100
h = 1
L = 5
n_replication = 1000000

def ses(alpha,n,D):
    s = 0
    for i in range(n-1):
        s += D[n-1-i]*(1-alpha)**i
    mu = alpha*s+D[0]*(1-alpha)**(n-1)
    return mu
#%% 经典方法 tc恒定
for n in [5, 10, 20, 100]:
    print("n={}".format(n))
    # 精确方法求解
    z_exact = []
    sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
    for i in range(n_replication):
        var_mu = sigma2_hat_1[i] * (alpha - alpha ** 2 + 2 * (1 - alpha) ** (2 * n)) / ((alpha - 2) * (alpha - 1))
        mu_hat_1 = mu_hat + sqrt(var_mu) * norm.rvs(size=1)
        z_exact.append(norm.rvs(loc=L * mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_exact = np.array(z_exact)
    S_exact = np.quantile(z_exact, b / (b + h))
 #   safety_exact = S_exact - np.mean(z_exact)
    z_exact_new = S_exact - z_exact
    TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                sum(b * z_exact_new[z_exact_new < 0])) / n_replication
    print(S_exact,  TC_exact, "exact stop")
    # 经典方法求解
    tc = []
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    z_classic_new = S_classic - z_exact
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic,  TC_classic, "classic stop")

    # 渐进方法求解
    z_asymtotic = []
    left_bound = -sqrt(n / 5)
    # left_bound = 0
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"),
                                                                              size=n_replication)
    mu_hat_1 = mu_hat + sqrt(sigma2_hat * alpha / (2 - alpha)) * norm.rvs(size=n_replication)
    for i in range(n_replication):
        z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                    scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_asymtotic = np.array(z_asymtotic)
    S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
    #safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
    z_asymtotic_new = S_asymtotic - z_exact
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    print(S_asymtotic,  TC_asymtotic, "asymtotic stop")

# 非参数方法
    print("n={}".format(n))
    # 非参数方法原始数据产生
    data = np.random.choice(z_exact.flatten(), n, replace = False)
    # 1. 普通quantile
    S_1 = np.quantile(data, q=b/(b+h))
    z_1 = S_1 - z_exact
    TC_1 = (sum(h * z_1[z_1 > 0]) -
                sum(b * z_1[z_1 < 0])) / n_replication
    print(S_1, TC_1, "method1 stop")
    # 2. bootstrap
    n_1, n_2 = int(np.floor(n * 0.5)), int(np.floor(n * 0.8))
    for n_choice in [n_1, n_2]:
        S = []
        for _ in range(1000):
            S.append(np.quantile(np.random.choice(data,n_choice,replace=True), q=b/(b+h)))
        S_2 = np.mean(S)
        z_2 = S_2 - z_exact
        TC_2 = (sum(h * z_2[z_2 > 0]) -
                sum(b * z_2[z_2 < 0])) / n_replication
        print(S_2, TC_2, "method2_{} stop".format(n_choice))
    # 3. SVP1-3
    data_ordered = np.sort(data)
    s_1 = 0
    p = b / (b + h)
    for i in range(1,n-1):
        s_1 += (binom.pmf(i+1, n, p)+binom.pmf(i, n, p)) / 2 * data_ordered[i]
    S_31 = (2 * binom.pmf(0, n, p) + binom.pmf(1, n, p)) / 2 * data_ordered[0] + binom.pmf(0, n, p) \
    / 2 * data_ordered[1] - binom.pmf(0, n, p) / 2 * data_ordered[2] + s_1 - binom.pmf(n, n, p) / 2 \
    * data_ordered[n - 3] + binom.pmf(n, n, p) / 2 * data_ordered[n - 2] + (2 * binom.pmf(n, n, p)  \
    + binom.pmf(n - 1, n, p)) / 2 * data_ordered[n-1]
    z_31 = S_31 - z_exact
    TC_31 = (sum(h * z_31[z_31 > 0]) - sum(b * z_31[z_31 < 0])) / n_replication
    print(S_31, TC_31, "method31 stop")

    s_2 = 0
    for i in range(n):
        s_2 = s_2 + binom.pmf(i, n, p) * data_ordered[i]
    S_32 = s_2 +(2 * data_ordered[n-1]-data_ordered[n-2]) * binom.pmf(n, n, p)
    z_32 = S_32 - z_exact
    TC_32 = (sum(h * z_32[z_32 > 0]) - sum(b * z_32[z_32 < 0])) / n_replication
    print(S_32, TC_32, "method32 stop")

    s_3 = 0
    for i in range(n):
        s_3 = s_3 + binom.pmf(i+1, n, p) * data_ordered[i]
    S_33 = s_3 +(2 * data_ordered[0]-data_ordered[1]) * binom.pmf(0, n, p)
    z_33 = S_33 - z_exact
    TC_33 = (sum(h * z_33[z_33 > 0]) - sum(b * z_33[z_33 < 0])) / n_replication
    print(S_33, TC_33, "method33 stop")
    to_be_continued = input("press anything to continue")






