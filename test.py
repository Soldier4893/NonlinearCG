import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(0)
np.seterr(all='ignore')

# initialization
n = 200
maxiter = 1000
solution = np.ones(n)
x0 = np.random.rand(n) * 10
a = np.random.rand(n,n)
A = a.T @ a + np.eye(n)
b = A @ solution
mode = 'f'

# plotting and algorithm
plt.figure(figsize=(12, 12))
i = 0
errors = np.array([0.5**i for i in range(n)])
plt.semilogy(errors[1:] - errors[:-1])


x = np.arange(1,n+1)
y = np.array([0.1**(i) for i in range(n)])
plt.semilogy(x,y, label = 'semilogy')
print(np.polyfit(np.log(x), np.log(y), 1))
# more plotting
plt.xlabel('Iteration')
plt.ylabel(f'{mode}-error (log scale)')
plt.title(f'{mode}-error vs Iteration')
plt.legend()
# plt.ylim([1e-12, f1(x0)])
plt.ylim([1e-12, 1])
plt.grid(True)
plt.show()
