import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import deco_print
from utils import h2T
from utils import poisson_initialization_n
from utils import random_initialization_n
from utils import greedy_alpha
from utils import obj_fun
from utils import update_tables
from utils import sample
from model import Parametric_Model
from model import get_h_and_loss_from_model
from model import train_model

### define parameters
N = 20
V_max = 50
I = 30
alpha = 0.001
n_iter = 100
###

### NN parameters
input_dim = 2
num_layer = 1
hidden_size_list = [5, 10, 20, 30]
###

case = 1

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


plt.figure('Figure 1')
plt.axhline(y=V_opt, xmin=0, xmax=V_max, color='black')

X_input = np.array([np.ones(V_max)]+[i*np.log(np.arange(V_max)+1) for i in range(1,input_dim)]).T
V_est = np.zeros((len(hidden_size_list), n_iter), dtype=int)

for k in range(len(hidden_size_list)):
	hidden_size = hidden_size_list[k]
	
	tables = [np.zeros((V_max,3), dtype=int) for _ in range(N)]

	models = dict()
	for i in range(N):
		deco_print('Creating model %d' %i, end='\r')
		models[i] = Parametric_Model('nn_%d_%d'%(k,i), V_max, input_dim=input_dim, num_layer=num_layer, hidden_size=hidden_size)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	h_hat = np.zeros((N, V_max))
	loss = np.zeros(N)
	
	for t in range(n_iter):
		for i in range(N):
			h_hat[i], loss[i] = get_h_and_loss_from_model(models[i], X_input, tables[i], sess)
		T_hat = h2T(h_hat)
		v_t, V_t = greedy_alpha(N, rho, T_hat, alpha)
		print(V_t)
		V_est[k,t] = V_t
		for _ in range(I):
			r_t = sample(N, rv, v_t)
			update_tables(N, tables, r_t, v_t)
		for i in range(N):
			loss[i] = train_model(models[i], X_input, tables[i], sess, loss[i])
	print('\n')

plt.plot(np.arange(n_iter), V_est[0], color='b', label='hidden size %d'%hidden_size_list[0])
plt.plot(np.arange(n_iter), V_est[1], color='r', label='hidden size %d'%hidden_size_list[1])
plt.plot(np.arange(n_iter), V_est[2], color='y', label='hidden size %d'%hidden_size_list[2])
plt.plot(np.arange(n_iter), V_est[3], color='m', label='hidden size %d'%hidden_size_list[3])
plt.xlabel('t')
plt.ylabel(r'Volume $\hat{V}^t$')
plt.legend()
plt.title(title)
plt.show()