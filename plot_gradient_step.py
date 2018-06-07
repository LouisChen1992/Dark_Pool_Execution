import numpy as np
import matplotlib.pyplot as plt

case = 1
title = 'Case I'

x = np.load('GradientSteps_%d.npz' %case)
nn = np.sum(x['nn'], axis=0)
logistic = np.sum(x['logistic'], axis=0)
n_iter = nn.shape[0]
nn_ave = np.cumsum(nn) / (np.arange(n_iter) + 1)
logistic_ave = np.cumsum(logistic) / (np.arange(n_iter) + 1)

plt.figure('steps')
plt.plot(np.arange(n_iter), nn, color='b', label='NN')
plt.plot(np.arange(n_iter), logistic, color='r', label='logistic')
plt.xlabel('t')
plt.ylabel('Gradient Steps')
plt.legend()
plt.title(title)

plt.figure('ave steps')
plt.semilogy(np.arange(n_iter), nn_ave, color='b', label='NN')
plt.semilogy(np.arange(n_iter), logistic_ave, color='r', label='logistic')
plt.xlabel('t')
plt.ylabel('Average Gradient Steps')
plt.legend()
plt.title(title)
plt.show()