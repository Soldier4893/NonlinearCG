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
    n = 100
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
    for beta in [AG, MAD, SD,PRP, HS, DY, N, FR]:
        i+= 1
        if beta != AG:
            x_final, errors, slope = NLCG(f1, g1, x0, beta, search, max_iter=maxiter, solution=solution, mode=mode)
        else:
            x_final, errors, slope = AG(f1, g1, x0, ag_search, max_iter=maxiter, solution=solution, mode=mode)
    
        plt.semilogy(errors, label=beta.__name__, linewidth = 2, linestyle="solid" if i < 5 else "dotted")
        print(f"Method: {beta.__name__:<5} Final Error: {np.format_float_scientific(errors[-1], precision=2):<10} Convergence Rate: {np.round(-slope[0], 4):<5}")

    # more plotting
    plt.xlabel('Iteration')
    plt.ylabel(f'{mode}-error (log scale)')
    plt.title(f'{mode}-error vs Iteration')
    plt.legend()
    plt.ylim([1e-12, f1(x0)])
    plt.grid(True)
    plt.show()