import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

df = pd.read_csv('pendulum_data.csv')
x = df['original'].to_numpy()
noixy_signal = df['noisy'].to_numpy()
t = df['t'].to_numpy()
print(df)
print(t)
lam = 1
data = df.to_numpy()
T = t.reshape(-1, 1)
UXX = rbf_kernel(T, T)
u = noixy_signal
UXX_inv_u = np.linalg.solve(UXX + lam**2*np.eye(101), u)
smoothed = UXX @ UXX_inv_u

plt.figure()
plt.plot(x, '-', label='original')
plt.plot(noixy_signal, ':', label='noisy')
plt.plot(smoothed, 'o', label='smoothed')
plt.legend()
plt.show()
