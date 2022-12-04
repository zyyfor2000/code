# %%初始参数设定
import math

from scipy.stats import norm, chi2, truncnorm, binom
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
for n in [5,10,20,100]:
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
    z_exact_new = S_exact - z_exact
    TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                sum(b * z_exact_new[z_exact_new < 0])) / n_replication
    print(S_exact, TC_exact, "exact stop")

    #  经典方法求解
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
    z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
    safety_classic = S_classic - np.mean(z_classic)
    z_classic_new = S_classic - z_exact
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                  sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic,  TC_classic, "classic stop")

    # 渐进方法求解
    z_asymtotic = []
    left_bound = -sqrt(n / 2)
    #left_bound = 0
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), loc=0, scale=1, size=n_replication)
    mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
    for i in range(n_replication):
        z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                    scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_asymtotic = np.array(z_asymtotic)
    S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
    safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
    z_asymtotic_new = S_asymtotic - z_exact
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    print(S_asymtotic,  TC_asymtotic, "asymtotic stop\n")
    # 非参数方法
    print("n={}".format(n))
    n_rep = 1000
    S_1_total = []
    TC_1_total = []
    S_21_total = []
    S_22_total = []
    TC_21_total = []
    TC_22_total = []
    S_31_total = []
    S_32_total = []
    S_33_total = []
    TC_31_total = []
    TC_32_total = []
    TC_33_total = []
    for _ in range(n_rep):
        # 非参数方法原始数据产生
        data = np.random.choice(z_exact.flatten(), n, replace = False)
        # 1. 普通quantile
        S_1 = np.quantile(data, q=b/(b+h))
        z_1 = S_1 - z_exact
        TC_1 = (sum(h * z_1[z_1 > 0]) -
                    sum(b * z_1[z_1 < 0])) / n_replication
        S_1_total.append(S_1)
        TC_1_total.append(TC_1)
        # print(S_1, TC_1, "method1 stop")
        # 2. bootstrap
        n_1, n_2 = int(np.floor(n * 0.5)), int(np.floor(n * 0.8))
        S = []
        for _ in range(1000):
            S.append(np.quantile(np.random.choice(data,n_1,replace=True), q=b/(b+h)))
        S_21 = np.mean(S)
        z_21 = S_21 - z_exact
        TC_21 = (sum(h * z_21[z_21 > 0]) -
                sum(b * z_21[z_21 < 0])) / n_replication
        S_21_total.append(S_21)
        TC_21_total.append(TC_21)
        S = []
        for _ in range(1000):
            S.append(np.quantile(np.random.choice(data,n_2,replace=True), q=b/(b+h)))
        S_22 = np.mean(S)
        z_22 = S_22 - z_exact
        TC_22 = (sum(h * z_22[z_22 > 0]) -
                sum(b * z_22[z_22 < 0])) / n_replication
        S_22_total.append(S_22)
        TC_22_total.append(TC_22)
        #print(S_2, TC_2, "method2_{} stop".format(n_choice))
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
        S_31_total.append(S_31)
        TC_31_total.append(TC_31)
        # print(S_31, TC_31, "method31 stop")

        s_2 = 0
        for i in range(n):
            s_2 = s_2 + binom.pmf(i, n, p) * data_ordered[i]
        S_32 = s_2 +(2 * data_ordered[n-1]-data_ordered[n-2]) * binom.pmf(n, n, p)
        z_32 = S_32 - z_exact
        TC_32 = (sum(h * z_32[z_32 > 0]) - sum(b * z_32[z_32 < 0])) / n_replication
        S_32_total.append(S_32)
        TC_32_total.append(TC_32)
        # print(S_32, TC_32, "method32 stop")

        s_3 = 0
        for i in range(n):
            s_3 = s_3 + binom.pmf(i+1, n, p) * data_ordered[i]
        S_33 = s_3 +(2 * data_ordered[0]-data_ordered[1]) * binom.pmf(0, n, p)
        z_33 = S_33 - z_exact
        TC_33 = (sum(h * z_33[z_33 > 0]) - sum(b * z_33[z_33 < 0])) / n_replication
        S_33_total.append(S_33)
        TC_33_total.append(TC_33)
        # print(S_33, TC_33, "method33 stop")
        #to_be_continued = input("press anything to continue")
    print(np.mean(S_1_total), np.mean(TC_1_total), "method1 stop")
    print(np.mean(S_21_total), np.mean(TC_21_total), "method2_{} stop".format(1))
    print(np.mean(S_22_total), np.mean(TC_22_total), "method2_{} stop".format(2))
    print(np.mean(S_31_total), np.mean(TC_31_total), "method31 stop")
    print(np.mean(S_32_total), np.mean(TC_32_total), "method32 stop")
    print(np.mean(S_33_total), np.mean(TC_33_total), "method33 stop")

#%% vilion图
from scipy.stats import norm, chi2, truncnorm, binom
from math import sqrt
import numpy as np
import pandas as pd

n_replication = 1000000

mu_hat = 10
sigma2_hat = 4
b = 20
h = 1
L = 5
n = 10
# 精确方法求解

z_exact = []
sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
for i in range(n_replication):
    mu_hat_1 = mu_hat + sqrt(sigma2_hat_1[i]) * norm.rvs(size=1) / sqrt(n)
    z_exact.append(norm.rvs(loc=L * mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_exact = np.array(z_exact)
S_exact = np.quantile(z_exact, b / (b + h))
safety_exact = S_exact - np.mean(z_exact)
z_exact_new = S_exact - z_exact
# TC_exact = h * z_exact_new[z_exact_new > 0] -b * z_exact_new[z_exact_new < 0]


#  经典方法求解
S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
eq1 = norm(loc=L * mu_hat, scale=sqrt(L * sigma2_hat))
z_classic = norm.rvs(L * mu_hat, sqrt(L * sigma2_hat), size=n_replication)
safety_classic = S_classic - np.mean(z_classic)
z_classic_new = S_classic - z_exact

# 渐进方法求解
z_asymtotic = []
left_bound = -sqrt(n / 2)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), loc=0, scale=1, size=n_replication)
mu_hat_1 = mu_hat + sqrt(sigma2_hat / n) * norm.rvs(size=n_replication)
for i in range(n_replication):
    z_asymtotic.append(norm.rvs(loc=L * mu_hat_1[i],
                                scale=sqrt(L * sigma2_hat_1[i]), size=1))
z_asymtotic = np.array(z_asymtotic)
S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
z_asymtotic_new = S_asymtotic - z_exact


#%%
df = pd.DataFrame()
df["z_exact_new"] = z_exact_new.flatten()
df["z_classic_new"] = z_classic_new.flatten()
df["z_asymtotic_new"] = z_asymtotic_new.flatten()
def fun(x):
    if x >= 0:
        return h*x
    else:
        return -b*x
# df["TC_classic"] = h*df[df["z_classic_new"]>0]["z_classic_new"]#+b*df[df["z_classic_new"]<0]["z_classic_new"]
# df["TC_classic"] = b*df[df["z_classic_new"]<0]["z_classic_new"]*(-1)
df["TC_classic"] = df["z_classic_new"].apply(lambda x: fun(x))
df["TC_exact"] = df["z_exact_new"].apply(lambda x: fun(x))
df["TC_asymtotic"] = df["z_asymtotic_new"].apply(lambda x: fun(x))

df["exact_classic"] = df["TC_exact"]-df["TC_classic"]
df["asymtotic_classic"] = df["TC_asymtotic"]-df["TC_classic"]
df["exact_asymtotic"] = df["TC_exact"]-df["TC_asymtotic"]
import seaborn as sns
import matplotlib.pyplot as plt
plt.violinplot(df[["exact_classic","asymtotic_classic","exact_asymtotic"]])
#sns.violinplot(y = df["exact_classic"])
plt.show(block = True)