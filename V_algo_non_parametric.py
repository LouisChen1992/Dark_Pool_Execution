import numpy as np
import matplotlib.pyplot as plt

from utils import deco_print
from utils import poisson_initialization_n
from utils import random_initialization_n
from utils import greedy
from utils import greedy_alpha
from utils import obj_fun
from utils import update_tables
from utils import KM_estimator_n
from utils import sample

### define parameters
N = 20
V_max = 300
I = 30
alpha = 0.001
n_iter = 50
###

case = 2

deco_print('There are %d dark pools. ' %N)

if case in [1,2]:
	if case == 1:
		lams = np.array([10]*N, dtype=int)
		rho = np.array([0.01]*N)
		Ks = [13, 15, 17, 19, 21]
		title = 'Case I'
	else:
		lams = np.array([5]*4+[10]*8+[20]*8, dtype=int)
		rho = np.array([0.01]*8+[0.002]*8+[0.001]*4)
		Ks = [15, 20, 25, 30, 40]
		title = 'Case II'
	### Case I & II
	rv, T, h = poisson_initialization_n(N, lams, V_max)
	deco_print('The maximum number of volume that can be executed in dark pools follow Poisson distribution with parameters ' + str(lams) + '. ')
elif case in [3,4]:
	if case == 3:
		hs = [np.array([0.1]*10+[0.2]*10+[0.15]*10)] * N
		rho = np.array([0.01]*N)
		Ks = [16, 18, 20, 22, 24]
		title = 'Case III'
	else:
		hs = np.random.rand(N, 30) * 0.2
		rho = np.random.rand(N) * 0.009 + 0.001
		Ks = [15, 20, 25, 30, 40]
		title = 'Case IV'
	### Case III & IV
	rv, T, h = random_initialization_n(N, hs, V_max)
	deco_print('The maximum number of volume that can be executed in dark pools follow a distribution specified by h. ')

f, _ = obj_fun(rho, T, V_max)
v_opt, V_opt = greedy_alpha(N, rho, T, alpha)
deco_print('The discount factors are ' + str(rho) + '. ')
deco_print('Optimal V: %d' %V_opt)
deco_print('Optimal allocation: ' + str(v_opt))


plt.figure('Figure 1')
plt.axhline(y=V_opt, xmin=0, xmax=V_max, color='black')

### Initialization 1
# tables = [np.zeros((V_max+1,3), dtype=int) for _ in range(N)]
# exploit = 1
# T_hat = np.zeros((N, V_max))
# T_hat[:, :exploit] = 1.0

### Initialization 2
# tables = [np.concatenate([np.arange(V_max+1)[::-1][:,np.newaxis]+1, \
# 	np.ones((V_max+1,1), dtype=int), \
# 	np.zeros((V_max+1,1), dtype=int)], axis=1) for _ in range(N)]
# h_hat, T_hat = KM_estimator_n(N, tables, V_max)
# exploit = False

### Initialization 3
"""
K = 0: Initialization 1
K = V_max + 1: Initialization 2
"""

for i_K in range(len(Ks)):
	K = Ks[i_K]

	tables = [np.concatenate([np.pad(np.arange(K)+1, (V_max-K+1,0), 'constant')[::-1][:,np.newaxis], \
		np.pad(np.ones(K, dtype=int), (0,V_max-K+1), 'constant')[:,np.newaxis], \
		np.zeros((V_max+1,1), dtype=int)], axis=1) for _ in range(N)]
	h_hat, T_hat = KM_estimator_n(N, tables, V_max)
	exploit = False

	V_est = []

	for t in range(n_iter):
		v_t, V_t = greedy_alpha(N, rho, T_hat, alpha)
		# f, f_star = obj_fun(rho, T_hat, V_max)
		# plt.plot(np.arange(V_max+1), f, color='r', linewidth = 2)
		print(V_t)
		V_est.append(V_t)
		for _ in range(I):
			r_t = sample(N, rv, v_t)
			update_tables(N, tables, r_t, v_t)
		h_hat, T_hat = KM_estimator_n(N, tables, V_max)
		if exploit:
			idx_exploit = np.sum(T_hat > 0, axis=1)
			for i in range(exploit):
				T_hat[range(N), idx_exploit+i] = T_hat[range(N), idx_exploit+i-1]

	print('\n')
	plt.plot(np.arange(n_iter), V_est, color=np.array([i_K, len(Ks)-1-i_K, 1.0])/(len(Ks)-1), label='K=%d' %K)

plt.xlabel('t')
plt.ylabel(r'$V^t$')
plt.legend()
plt.title(title)
plt.show()
