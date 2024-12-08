import numpy as np
from nonlincg import *
import matplotlib.pyplot as plt

# linear case
def f1(x):
    return 1/2*x@A@x-b@x
def g1(x):
    return A@x-b

if __name__ == "__main__":
    # np.random.seed(0)
    np.seterr(all='ignore')

    # initialization
    n = 200
    maxiter = 2000
    solution = np.ones(n)
    x0 = np.random.rand(n) * 10
    a = np.random.rand(n,n)
    A = a.T @ a + np.eye(n)
    b = A @ solution
    mode = 'f'

    # plotting and algorithm
    plt.figure(figsize=(12, 12))
    i = 0
    for beta in [AG, SD,PRP, MAD, HS, DY, N, FR, CD]:
        i+= 1
        if beta != AG:
            x_final, errors, (coeff, isSuperlinear) = NLCG(f1, g1, x0, beta, max_iter=maxiter, solution=solution, mode=mode, A = A)
        else:
            x_final, errors, (coeff, isSuperlinear) = AG(f1, g1, x0, max_iter=maxiter, solution=solution, mode=mode)
        plt.semilogy(errors, label=beta.__name__, linewidth = 2, linestyle="solid" if i < 5 else "dashed")
        convergence = 'Least Squares Failed'
        if isSuperlinear: convergence = f'Linear, order {coeff[0]:<5.3f} rate {coeff[1]:<5.3f}' 
        elif isSuperlinear == False: convergence = f'Sublinear     1/k^{coeff[0]:<5.3f}'
        print(f"Method: {beta.__name__:<5} Final Error: {errors[-1]:<10.2e}{convergence}")
        
    x = np.arange(1,n+1)
    y = np.array([0.1**(i) for i in range(n)])
    # more plotting
    plt.xlabel('Iteration')
    plt.ylabel(f'{mode}-error (log scale)')
    plt.title(f'{mode}-error vs Iteration')
    plt.legend()
    plt.ylim([1e-12, f1(x0)])
    plt.grid(True)
    plt.show()
    