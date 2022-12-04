# %%初始参数设定
from scipy.stats import norm, chi2, truncnorm
from math import sqrt
import numpy as np
alpha_hat = 10
beta_hat = 1
sigma2_hat = 4
n = 5  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000

#%% 经典方法求解，tc恒为定值
for n_change in [5, 10, 20, 100]:
    S_classic = L * (alpha_hat) + 0.5 * (L ** 2 + 2 * n_change * L + L) * beta_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    z_classic = 0
    for i in range(L):
        z_classic = z_classic + alpha_hat + (n_change+i+1)*beta_hat+norm.rvs(0, sqrt(sigma2_hat), size=n_replication)
    z_classic_new = S_classic - z_classic
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                  sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic,  TC_classic, "n_change = {}".format(n_change))
#%% 经典方法改进 s不变
n_replication_c = 1000
for n_change in [5, 10, 20, 100]:
    tc = []
    S_classic = L * (alpha_hat) + 0.5 * (L ** 2 + 2 * n_change * L + L) * beta_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    data_in_practice = np.zeros((n_replication_c,n_change))
    t = [i+1 for i in list(range(n_change))]
    for i in range(n_change):
        data_in_practice[:,i]= alpha_hat+(i+1)*beta_hat+ norm.rvs(0, sqrt(sigma2_hat), size=n_replication_c)
    for i in range(n_replication_c):
        line = np.polyfit(t,data_in_practice[i].tolist(),deg=1,full=True)
        alpha_hat_hat, beta_hat_hat, sigma2_hat_hat = line[0][1], line[0][0], line[1]/(n_change-1)
        z_classic = norm.rvs(L * (alpha_hat_hat) + 0.5 * (L ** 2 + 2 * n_change * L + L), sqrt(L*sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
    print(S_classic,  np.mean(tc), "n_change = {}".format(n_change))

#%% 经典方法改进 s变
n_replication_c = 10000
for n_change in [5, 10, 20, 100]:
    s_total = []
    tc = []
    alpha_hat_hat, beta_hat_hat,sigma2_hat_hat = alpha_hat,beta_hat,sigma2_hat
    data_in_practice = np.zeros((n_replication_c,n_change))
    t = [i+1 for i in list(range(n_change))]
    for i in range(n_change):
        data_in_practice[:,i]= alpha_hat+(i+1)*beta_hat+ norm.rvs(0, sqrt(sigma2_hat), size=n_replication_c)
    for i in range(n_replication_c):
        line = np.polyfit(t,data_in_practice[i].tolist(),deg=1,full=True)
        alpha_hat_hat, beta_hat_hat, sigma2_hat_hat = line[0][1], line[0][0], line[1]/(n_change-1)
        S_classic = L * (alpha_hat_hat) + 0.5 * (L ** 2 + 2 * n_change * L + L) * beta_hat_hat + sqrt(L) * (
            -norm.isf(q=b / (b + h))) * sqrt(sigma2_hat_hat)
        s_total.append(S_classic)
        z_classic = norm.rvs(L * alpha_hat_hat + 0.5 * (L ** 2 + 2 * n_change * L + L)* beta_hat_hat, sqrt(L*sigma2_hat_hat), size=n_replication)
        z_classic_new = S_classic - z_classic
        TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
        tc.append(TC_classic)
    print(np.mean(s_total),  np.mean(tc), "n_change = {}".format(n_change))

# %% 精确方法求解
for n in [5, 10, 20, 100]:
    z_exact = []
    mu_hat = L * (alpha_hat) + 0.5 * (L ** 2 + 2 * n * L + L) * beta_hat
    sigma2_hat_1 = (n - 2) * sigma2_hat / chi2.rvs(n - 2, size=n_replication)
    for i in range(n_replication):
        v = sigma2_hat_1[i]*np.array([[(4*n+2)/(n*(n-1)),-6/(n*(n-1))],[-6/(n*(n-1)),12/(n*(n**2-1))]])
        var_mu = np.array([L,0.5*(L**2+2*n*L+L)])@v@np.array([[L],[0.5*(L**2+2*n*L+L)]])
        mu_hat_1 = mu_hat + sqrt(var_mu) * norm.rvs(size=1)
        z_exact.append(norm.rvs(loc=mu_hat_1, scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_exact = np.array(z_exact)
    S_exact = np.quantile(z_exact, b / (b + h))
    safety_exact = S_exact - np.mean(z_exact)
    z_exact_new = S_exact - z_exact
    TC_exact = (sum(h * z_exact_new[z_exact_new > 0]) -
                sum(b * z_exact_new[z_exact_new < 0])) / n_replication
    print(S_exact, safety_exact, TC_exact, "n_change = {}".format(n),"exact stop")

# %%渐进方法求解
for n in [5, 10, 20, 100]:
    z_asymtotic = []
    left_bound = -sqrt(n / 2)
    #left_bound = 0
    mu_hat = L * (alpha_hat) + 0.5 * (L ** 2 + 2 * n * L + L) * beta_hat
    sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
    v = sigma2_hat * np.array(
        [[(4 * n + 2) / (n * (n - 1)), -6 / (n * (n - 1))], [-6 / (n * (n - 1)), 12 / (n * (n ** 2 - 1))]])
    var_mu = np.array([L, 0.5 * (L ** 2 + 2 * n * L + L)]) @ v @ np.array([[L], [0.5 * (L ** 2 + 2 * n * L + L)]])
    mu_hat_1 = mu_hat + sqrt(var_mu) * norm.rvs(size=n_replication)
    for i in range(n_replication):
        z_asymtotic.append(norm.rvs(loc=mu_hat_1[i],
                                    scale=sqrt(L * sigma2_hat_1[i]), size=1))
    z_asymtotic = np.array(z_asymtotic)
    S_asymtotic = np.quantile(z_asymtotic, b / (b + h))
    safety_asymtotic = S_asymtotic - np.mean(z_asymtotic)
    z_asymtotic_new = S_asymtotic - z_asymtotic
    TC_asymtotic = (sum(h * z_asymtotic_new[z_asymtotic_new > 0]) -
                    sum(b * z_asymtotic_new[z_asymtotic_new < 0])) / n_replication
    print(S_asymtotic, safety_asymtotic, TC_asymtotic, "n_change = {}".format(n),"asymtotic stop")





