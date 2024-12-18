import numpy as np
from nonlincg import *
import matplotlib.pyplot as plt
    
def calculations(i):
    # initialization
    index = i
    n = 600
    maxiter = 2000
    x0 = 1.5 * np.ones(n)
    f, g, solution = get_objective_function(n, index = index) # index = 0, 1, 2, 3, 4, 5
    mode = 'f'

    # plotting and algorithm
    i = 0
    e0 = 0
    plt.figure(figsize=(12, 12))
    for beta in [AG, SD, FR, PRP, HS, DY, LS, N, MPRP]:
        i+= 1
        if beta != AG:
            x_final, errors, (coeff, isSuperlinear) = NLCG(f, g, x0, beta, max_iter=maxiter, solution=solution, mode=mode)
        else:
            x_final, errors, (coeff, isSuperlinear) = AG(f, g, x0, max_iter=maxiter, solution=solution, mode=mode)
        
        e0 = errors[0]
        plt.semilogy(errors, label=beta.__name__, linewidth = 2, linestyle="solid" if i < 7 else "dashed" if i < 6 else "dashdot")
        
        convergence = 'Least Squares Failed or Did not converge'
        if isSuperlinear: convergence = f'Linear, order {coeff[0]:<5.3f} rate {coeff[1]:<5.3f}' 
        elif isSuperlinear == False: convergence = f'Sublinear     1/k^{coeff[0]:<5.3f}'
        print(f"Method: {beta.__name__:<5} Final Error: {errors[-1]:<10.2e}{convergence}")
        plt.xlabel('Iteration')
        plt.ylabel(f'{mode}-error (log scale)')
        plt.title(f'{mode}-error vs Iteration Number, problem {index}')
        plt.legend()
        plt.ylim(top=e0, bottom=1e-10)
        plt.grid(True)
        plt.savefig(f'{mode}_{index}_{n}.png', dpi=300)
    
if __name__ == "__main__":
    np.random.seed(574)
    np.seterr('ignore')
    
    for i in range(4):
        calculations(i)
    
    #plt.semilogy([e0*0.9**(i**2) for i in range(maxiter)], label='test', linewidth = 2, linestyle="solid" if i < 5 else "dashed")
    
    plt.show()