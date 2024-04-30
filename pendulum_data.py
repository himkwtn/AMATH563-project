import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

x0 = 0.7
xN = x0
t0 = 0
tN = 2 * np.pi
N = 101
t = np.linspace(t0, tN, N)
h = t[1] - t[0]


def F(x):
    y = np.zeros((N-2, 1))
    for i in range(N-2):
        y[i] = x[i] - 2*x[i+1] + x[i+2] + h**2*np.sin(x[i+1])
    return y


def J(x):
    y = np.zeros((N-2, N-2))

    for i in range(N-2):
        y[i, i] = -2 + h**2*np.cos(x[i+1])
        if (i > 1):
            y[i, i-1] = 1
        if (i < N-3):
            y[i, i+1] = 1

    return y


max_steps = 500
x = 0.7 * np.cos(t) + 0.5 * np.sin(t)
k = 0
X = [x]

while np.max(np.abs(F(x))) >= 1e-8 and k < max_steps:
    dx = np.linalg.solve(J(x), F(x))
    new_x = np.zeros_like(x)
    new_x[0] = x0
    new_x[-1] = xN
    new_x[1:-1] = x[1:-1] - dx[:, 0]
    x = new_x
    k += 1
    X.append(new_x)

noise = np.random.normal(0, 0.1, 101)
noisy_data = x + noise

df = pd.DataFrame(np.array([t, x, noisy_data]).T,
                  columns=['t', 'original', 'noisy'])
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

df.to_csv(f'pendulum_data_{timestamp}.csv', index=False)

plt.figure()
plt.plot(t, x, '-', label=f'true data')
plt.plot(t, noisy_data, ':', label=f'noisy data')

plt.legend()
plt.title(r'$\theta^{(0)} = 0.7\cos(t) + 0.5 \sin(t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.show()
