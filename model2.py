# %%初始参数设定
from scipy.stats import norm, chi2, truncnorm
from math import sqrt
import numpy as np
mu_hat = 10
sigma2_hat = 4
alpha = 0.2
n = 5  # 5,10,20,100
b = 20
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
for n_change in [5, 10, 20, 100]:
    tc = []
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    z_classic = norm.rvs(L*mu_hat, sqrt(L*sigma2_hat), size=n_replication)
    z_classic_new = S_classic - z_classic
    TC_classic = (sum(h * z_classic_new[z_classic_new > 0]) -
                      sum(b * z_classic_new[z_classic_new < 0])) / n_replication
    print(S_classic,  TC_classic, "n_change = {}".format(n_change))

#%% 经典方法改进 s不变
for n_change in [5, 10, 20, 100]:
    tc = []
    S_classic = L * mu_hat + sqrt(L) * (-norm.isf(q=b / (b + h))) * sqrt(sigma2_hat)
    for i in range(1000):
        data_in_practice = norm.rvs(mu_hat, sqrt(sigma2_hat), size=n_change)
        mu_hat_hat = ses(alpha,n,data_in_practice)
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
        mu_hat_hat = ses(alpha,n,data_in_practice)
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
n = 100  # 5,10,20,100
b = 20
h = 1
L = 5
z_exact = []
sigma2_hat_1 = (n - 1) * sigma2_hat / chi2.rvs(n - 1, size=n_replication)
for i in range(n_replication):
    var_mu = sigma2_hat_1[i]*(alpha-alpha**2+2*(1-alpha)**(2*n))/((alpha-2)*(alpha-1))
    mu_hat_1 = mu_hat + sqrt(var_mu) * norm.rvs(size=1)
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
n = 100  # 5,10,20,100
b = 20
h = 1
L = 5
n_replication = 1000000
z_asymtotic = []
left_bound = -sqrt(n / 2)
#left_bound = 0
sigma2_hat_1 = sigma2_hat + sqrt(2 * sigma2_hat ** 2 / n) * truncnorm.rvs(left_bound, float("inf"), size=n_replication)
mu_hat_1 = mu_hat + sqrt(sigma2_hat * alpha/ (2-alpha)) * norm.rvs(size=n_replication)
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

