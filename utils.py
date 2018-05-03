import numpy as np
from scipy.stats import poisson

class RandomVariable:
	def __init__(self, h):
		self._h = h
		self._s_max = len(h)
		self._T = h2T(np.array([h]))[0]

	def rvs(self):
		s = 0
		while s < self._s_max:
			if np.random.rand() < self._h[s]:
				return s
			else:
				s += 1
		return self._s_max

	def cdf(self, x):
		if x >= self._s_max:
			return 1.0
		else:
			return 1.0-self._T[x]


def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)

def poisson_initialization_n(N, lams, V_max):
	"""
	Parameters
	----------
	N: int
		number of dark pools
	lams: list
		poission parameters lambda
	V_max: int
		upper limit

	Returns
	----------
	rv: list of poission random variables
	T: N * V_max
		tail distribution for N poisson random variables
		T[i,j] = P(X_i >= j+1), j=0,...,V_max-1
	h: N * (V_max - 1)
		conditional distributions
		h[i,j] = P(X_i = j|X_i >= j), j=0,...,V_max-1
	"""
	assert(N == len(lams))
	rv = [poisson(lam) for lam in lams]
	T = np.zeros((N, V_max))
	for i in range(N):
		for j in range(V_max):
			T[i,j] = 1 - rv[i].cdf(j)
	h = T2h(T)
	return rv, T, h

def poisson_initialization(lam, V_max):
	rvs, Ts, hs = poisson_initialization_n(1, [lam], V_max)
	return rvs[0], Ts[0], hs[0]

def random_initialization_n(N, hs, V_max):
	assert(N == len(hs))
	rv = [RandomVariable(hs[i]) for i in range(N)]
	T = np.zeros((N, V_max))
	for i in range(N):
		for j in range(V_max):
			T[i,j] = 1 - rv[i].cdf(j)
	h = T2h(T)
	return rv, T, h

def random_initialization(h, V_max):
	rvs, Ts, hs = random_initialization_n(1, [h], V_max)
	return rvs[0], Ts[0], hs[0]

def greedy(N, rho, T, V):
	T_rho = T * rho[:,np.newaxis]
	v = np.zeros(N, dtype=int)
	for _ in range(V):
		p = np.random.permutation(N)
		temp = T_rho[np.arange(N), v]
		idx = p[np.argmax(temp[p])]
		v[idx] += 1
	return v

def greedy_alpha(N, rho, T, alpha):
	T_rho = T * rho[:,np.newaxis]
	v = np.zeros(N, dtype=int)
	while max(T_rho[np.arange(N), v]) > alpha:
		p = np.random.permutation(N)
		temp = T_rho[np.arange(N), v]
		idx = p[np.argmax(temp[p])]
		v[idx] += 1
	V = np.sum(v)
	return v, V

def obj_fun(rho, T, V):
	T_rho = T * rho[:,np.newaxis]
	T_rho_reshape = T_rho.reshape(-1)
	T_rho_sorted = sorted(T_rho_reshape, key=lambda x:-x)
	return np.concatenate(([0],np.cumsum(T_rho_sorted[:V]))), np.sum(T_rho)

def T2h(T):
	h = np.zeros(T.shape)
	for i in range(T.shape[0]):
		h[i,0] = 1 - T[i,0]
		for j in range(1,T.shape[1]):
			h[i,j] = 1 - T[i,j] / T[i,j-1]
	return h

def h2T(h):
	T = np.zeros(h.shape)
	for i in range(h.shape[0]):
		T[i,0] = 1 - h[i,0]
		for j in range(1, h.shape[1]):
			T[i,j] = T[i,j-1] * (1 - h[i,j])
	return T

def update_table(table, r, v):
	"""
	Parameters
	----------
	table: V_max * 3, int
		survival table
	r, v: new observations
	"""
	table[:(r+1),0] += 1
	if r < v:
		table[r,1] += 1
	else:
		table[r,2] += 1

def update_tables(N, tables, rs, vs):
	for i in range(N):
		update_table(tables[i], rs[i], vs[i])

def KM_estimator(table):
	d = table[:-1,1]
	Y = table[1:,0]
	h = d / (d + Y)
	h[np.isnan(h)] = 1.0 # remove invalid number
	return h, h2T(h[np.newaxis,:])[0]

def KM_estimator_n(N, tables, V_max):
	h = np.zeros((N, V_max))
	T = np.zeros((N, V_max))
	for i in range(N):
		h[i,:], T[i,:] = KM_estimator(tables[i])
	return h, T

def sample(N, rv, v):
	D = np.zeros(N, dtype=int)
	for i in range(N):
		D[i] = rv[i].rvs()
	return np.minimum(D, v)