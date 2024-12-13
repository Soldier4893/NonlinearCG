import numpy as np
from nonlincg import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(574)
    np.seterr(all='ignore')

    # initialization
    n = 20
    maxiter = 500
    x0 = np.random.rand(n) * 10

    f, g, solution = get_objective_function(n, index = 0) # index = 0, 1, 2
    mode = 'f'

    # plotting and algorithm
    i = 0
    e0 = 0
    plt.figure(figsize=(12, 12))
    for beta in [AG, SD,PRP, HS, DY, N, FR, LS]:
        i+= 1
        if beta != AG:
            x_final, errors, (coeff, isSuperlinear) = NLCG(f, g, x0, beta, max_iter=maxiter, solution=solution, mode=mode)
        else:
            x_final, errors, (coeff, isSuperlinear) = AG(f, g, x0, max_iter=maxiter, solution=solution, mode=mode)
        
        e0 = errors[0]
        plt.semilogy(errors, label=beta.__name__, linewidth = 2, linestyle="solid" if i < 3 else "dashed" if i < 6 else "dashdot")
        
        convergence = 'Least Squares Failed or Did not converge'
        if isSuperlinear: convergence = f'Linear, order {coeff[0]:<5.3f} rate {coeff[1]:<5.3f}' 
        elif isSuperlinear == False: convergence = f'Sublinear     1/k^{coeff[0]:<5.3f}'
        print(f"Method: {beta.__name__:<5} Final Error: {errors[-1]:<10.2e}{convergence}")
    

    # plt.semilogy([np.linalg.norm(e0)*0.851**(i**0.995) for i in range(300)], label='test', linewidth = 2, linestyle="solid" if i < 5 else "dashed")
    # more plotting
    plt.xlabel('Iteration')
    plt.ylabel(f'{mode}-error (log scale)')
    plt.title(f'{mode}-error vs Iteration Number')
    plt.legend()
    plt.ylim([1e-12, e0])
    plt.grid(True)
    plt.show()
    