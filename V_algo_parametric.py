import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import deco_print
from utils import h2T
from utils import poisson_initialization_n
from utils import zipf_initialization_n
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
n_iter = 1000 #100
###

### NN parameters
input_dim = 2
num_layer = 1
hidden_size = 30
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
elif case in [5,6]:
	if case == 5:
		alphas = np.array([1.2]*N)
		rho = np.array([0.002]*N)
		title = 'Case V'
	else:
		alphas = np.array([1.2]*4+[1.5]*8+[2.0]*8)
		rho = np.array([0.002]*8+[0.005]*8+[0.01]*4)
		title = 'Case VI'
	### Case V & VI
	rv, T, h = zipf_initialization_n(N, alphas, V_max)
	deco_print('The maximum number of volume that can be executed in dark pools follow Zipf\'s distribution with parameters ' + str(alphas) + '. ')

f, _ = obj_fun(rho, T, V_max)
v_opt, V_opt = greedy_alpha(N, rho, T, alpha)
deco_print('The discount factors are ' + str(rho) + '. ')
deco_print('Optimal V: %d' %V_opt)
deco_print('Optimal allocation: ' + str(v_opt))


plt.figure('Figure 1')
plt.axhline(y=V_opt, xmin=0, xmax=V_max, color='black')

tables_nn = [np.zeros((V_max+1,3), dtype=int) for _ in range(N)]
tables_logistic = [np.zeros((V_max+1,3), dtype=int) for _ in range(N)]

nn_models = dict()
logistic_models = dict()
for i in range(N):
	deco_print('Creating model %d' %i, end='\r')
	nn_models[i] = Parametric_Model('nn%d'%i, V_max+1, input_dim=input_dim, num_layer=num_layer, hidden_size=hidden_size)
	logistic_models[i] = Parametric_Model('logistic%d'%i, V_max+1, input_dim=input_dim, num_layer=0, hidden_size=0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

X_input = np.array([np.ones(V_max+1)]+[i*np.log(np.arange(V_max+1)+1) for i in range(1,input_dim)]).T

V_est_nn = []
V_est_logistic = []

h_hat_nn = np.zeros((N, V_max+1))
loss_nn = np.zeros(N)
h_hat_logistic = np.zeros((N, V_max+1))
loss_logistic = np.zeros(N)
	
for t in range(n_iter):
	for i in range(N):
		h_hat_nn[i], loss_nn[i] = get_h_and_loss_from_model(nn_models[i], X_input, tables_nn[i], sess)
		h_hat_logistic[i], loss_logistic[i] = get_h_and_loss_from_model(logistic_models[i], X_input, tables_logistic[i], sess)
	T_hat_nn = h2T(h_hat_nn)
	T_hat_logistic = h2T(h_hat_logistic)
	v_t_nn, V_t_nn = greedy_alpha(N, rho, T_hat_nn[:,:-1], alpha)
	v_t_logistic, V_t_logistic = greedy_alpha(N, rho, T_hat_logistic[:,:-1], alpha)
	print(V_t_nn)
	print(V_t_logistic)
	V_est_nn.append(V_t_nn)
	V_est_logistic.append(V_t_logistic)
	for _ in range(I):
		r_t_nn = sample(N, rv, v_t_nn)
		r_t_logistic = sample(N, rv, v_t_logistic)
		update_tables(N, tables_nn, r_t_nn, v_t_nn)
		update_tables(N, tables_logistic, r_t_logistic, v_t_logistic)
	for i in range(N):
		loss_nn[i] = train_model(nn_models[i], X_input, tables_nn[i], sess, loss_nn[i])
		loss_logistic[i] = train_model(logistic_models[i], X_input, tables_logistic[i], sess, loss_logistic[i])
	print('\n')

plt.plot(np.arange(n_iter), V_est_nn, color='b', label='NN')
plt.plot(np.arange(n_iter), V_est_logistic, color='r', label='logistic')
plt.xlabel('t')
plt.ylabel(r'Volume $\hat{V}^t$')
plt.legend()
plt.title(title)
plt.show()