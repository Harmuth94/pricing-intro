import numpy.random as npr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

npr.seed(12345)

par_lambda = 0.3

gamma_shape = 20
gamma_scale = 300

premium_naive = par_lambda * gamma_shape * gamma_scale

n_iter = 10000
n_policy = 10000
sims = []
single_policy = []
for i in range(n_iter):
    n_claims = npr.poisson(par_lambda, n_policy)
    severity = npr.gamma(gamma_shape, gamma_scale, n_claims.sum())
    # Compute the average claim expense
    sims.append(severity.sum()/n_policy)

# Plot the results as histogram. Add a vertical line for the theoretical value
sns.distplot(sims, kde=False)
plt.axvline(premium_naive, color='red')
plt.savefig('naive_premium_dist.png', dpi=300)
plt.close()
# Cummulative profit
cum_profit = -np.cumsum(sims)*n_policy + premium_naive*np.arange(1, n_iter+1)*n_policy

# Plot the results as line
plt.plot(cum_profit)
plt.savefig('naive_profit_cum.png', dpi=300)
plt.close()

# load premium with 5% safety margin
premium_safety = premium_naive * 1.01

# Cummulative profit
cum_profit_safety =  premium_safety*np.arange(1, n_iter+1)*n_policy-np.cumsum(sims)*n_policy
plt.plot(cum_profit_safety)
plt.savefig('loaded_profit_cum.png', dpi=300)
plt.close()



# Single policy
n_iter_single = 10000000

single_policy = []

n_claims = npr.poisson(par_lambda, n_iter_single)
severity = npr.gamma(gamma_shape, gamma_scale, n_claims.sum())
# Compute the average claim expense
sev_idx = 0
for n_claim in n_claims:
    sev = 0
    for i in range(n_claim):
        sev += severity[sev_idx]
        sev_idx += 1
    single_policy.append(sev)    


# Plot the results as histogram. Add a vertical line for the theoretical value
sns.distplot(single_policy, kde=False)
plt.xlim(0, 15000)
plt.axvline(premium_naive, color='red')
plt.savefig('single_policy_dist.png', dpi=300)
plt.close()

