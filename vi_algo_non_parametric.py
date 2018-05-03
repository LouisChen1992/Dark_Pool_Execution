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
K = 20
n_iter = 20
n_simulation = 50
###

###
case = 1
###

deco_print('There are %d dark pools. ' %N)

if case in [1,2]:
	if case == 1:
		lams = np.array([10]*N, dtype=int)
		rho = np.array([0.01]*N)
		title = 'Case I'
	else:
		lams = np.array([5]*4+[10]*8+[20]*8, dtype=int)
		rho = np.array([0.01]*8+[0.002]*8+[0.001]*4)
		title = 'Case II'
	### Case I & II
	rv, T, h = poisson_initialization_n(N, lams, V_max)
	deco_print('The maximum number of volume that can be executed in dark pools follow Poisson distribution with parameters ' + str(lams) + '. ')
elif case in [3,4]:
	if case == 3:
		hs = [np.array([0.1]*10+[0.2]*10+[0.15]*10)] * N
		rho = np.array([0.01]*N)
		title = 'Case III'
	else:
		hs = np.random.rand(N, 30) * 0.2
		rho = np.random.rand(N) * 0.009 + 0.001
		title = 'Case IV'
	### Case III & IV
	rv, T, h = random_initialization_n(N, hs, V_max)
	deco_print('The maximum number of volume that can be executed in dark pools follow a distribution specified by h. ')

f, _ = obj_fun(rho, T, V_max)
v_opt, V_opt = greedy_alpha(N, rho, T, alpha)
deco_print('The discount factors are ' + str(rho) + '. ')
deco_print('Optimal V: %d' %V_opt)
deco_print('Optimal allocation: ' + str(v_opt))


### Output v
v_out = np.zeros((n_simulation, N))
###

for k in range(n_simulation):
	tables = [np.concatenate([np.pad(np.arange(K)+1, (V_max-K+1,0), 'constant')[::-1][:,np.newaxis], \
		np.pad(np.ones(K, dtype=int), (0,V_max-K+1), 'constant')[:,np.newaxis], \
		np.zeros((V_max+1,1), dtype=int)], axis=1) for _ in range(N)]
	h_hat, T_hat = KM_estimator_n(N, tables, V_max)
	exploit = False

	for t in range(n_iter):
		v_t, V_t = greedy_alpha(N, rho, T_hat, alpha)
		for _ in range(I):
			r_t = sample(N, rv, v_t)
			update_tables(N, tables, r_t, v_t)
		h_hat, T_hat = KM_estimator_n(N, tables, V_max)
		if exploit:
			idx_exploit = np.sum(T_hat > 0, axis=1)
			for i in range(exploit):
				T_hat[range(N), idx_exploit+i] = T_hat[range(N), idx_exploit+i-1]

	v_out[k] = v_t

ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(ind, v_opt, width, color='r')
rects2 = ax.bar(ind+width, np.mean(v_out,0), width, color='y', yerr=np.std(v_out,0))
ax.set_xlabel('Dark Pool')
ax.set_ylabel(r'Allocation $\hat{v}_i$')
ax.set_xticks(ind+width/2)
ax.set_xticklabels(tuple(str(i) for i in ind))
ax.legend((rects1[0], rects2[0]), ('Opt', 'Alg'), loc='best')
plt.title(title)

def autolabel(rects):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
			'%.0f' % float(height),
			ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()