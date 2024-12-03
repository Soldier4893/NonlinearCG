import math
from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Backtracking line search with Wolfe conditions
def line_search(f, grad_f, x, d, alpha_init=10.0, rho=0.5, c1=1e-4, c2=0.9):
    alpha = alpha_init
    grad = grad_f(x)
    
    # Check Wolfe conditions iteratively, reduce alpha until both conditions hold
    for _ in range(100):
        # Armijo (sufficient decrease) condition
        if f(x + alpha * d) <= f(x) + c1 * alpha * np.dot(grad, d):
            # Curvature (strong Wolfe) condition
            if np.dot(grad_f(x + alpha * d), d) >= c2 * np.dot(grad, d):
                break  # Both conditions satisfied
        alpha *= rho  # Reduce alpha if conditions are not satisfied
    
    return alpha



# Nonlinear Conjugate Gradient Method
def NLCG(f, grad_f, x0, cb=None, ls=line_search, max_iter=1000, solution=None, mode = 0):
    def error_func(mode):
        if mode == 0:
            return np.linalg.norm(f(x) - f(solution))
        elif mode == 1:
            return np.linalg.norm(grad_f(x))
        else:
            return np.linalg.norm(x - solution)
    tol = {0: 1e-12, 1: 1e-7, 2: 1e-8}[mode]
    errors = tol * np.ones(max_iter)
    x = x0
    g = grad_f(x)
    d = -g
    counter = 1
    
    

    for k in range(max_iter):
        if error_func(mode) < tol:
            break
        alpha = ls(f, grad_f, x, d)
        
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        
        beta = cb(g, g_new, d, x, x_new, k)
        d_new = -g_new + beta * d
        
        errors[k] = error_func(mode)

        x = x_new
        g = g_new
        d = d_new
        counter += 1
    C = np.log(errors[0:counter-1])
    D = np.arange(1,counter)
    # print(C.shape, D.shape)
    coef = np.polyfit(D, C, 1)
    return x, errors, coef

def FR(g, g_new, d, x, x_new, k):
    return np.linalg.norm(g_new)**2/np.linalg.norm(g)**2
def PRP(g, g_new, d, x, x_new, k):
    return np.dot(g_new, g_new-g) / np.linalg.norm(g)
def HS(g, g_new, d, x, x_new, k):
    return np.dot(g_new, g_new - g)/np.dot(d, g_new - g)
def DY(g, g_new, d, x, x_new, k):
    return np.dot(g_new, g_new)/np.dot(d, g_new-g)
def N(g, g_new, d, x, x_new, k):
    a = g_new - g - 2*np.dot(g_new-g, g_new-g)/np.dot(d, g_new-g)
    b = g_new / np.dot(d, g_new - g)
    return np.dot(a,b)
def LS(g, g_new, d, x, x_new, k):
    return -np.dot(g_new, g_new-g)/np.dot(d, g)
def CD(g, g_new, d, x, x_new, k):
    return -np.dot(g_new, g_new)/np.dot(d,g)
def MAD(g, g_new, d, x, x_new, k):
    mu = 0.9
    mu2 = mu**2
    eta = np.dot(d,g_new)
    zeta = np.dot(d,d)*np.dot(g_new,g_new)
    term = (eta-mu*zeta)*(eta+mu*zeta)
    beta = 0

    l = 2*np.random.randint(0,2)-1
    if eta == 0:
        beta = l*np.sqrt(abs(1-mu2))*np.dot(g_new, g_new)/(mu*zeta)
    elif term == 0:
        beta = np.dot(g_new, g_new)/(2*eta)
    else:
        numer = (1-mu2)*eta+ l*mu*zeta*np.sqrt(abs((1-mu2)*(1-eta/zeta)*(1+eta/zeta)))
        beta = np.dot(g_new, g_new)*numer/term
    return np.random.random() * beta
def SD(g, g_new, d, x, x_new, k):
    return 0
def linear_f(x):
    return 1/2*x@A@x-b@x
def g_linear_f(x):
    return A@x-b
def rosenbrock(x):
    m = len(x)
    sum = 0
    for i in range(m-1):
        sum += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return sum
def rosenbrock_grad(x):
    m = len(x)
    grad = np.zeros(n)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1)
    for i in range(1, m-1):
        grad[i] = -400 * x[i] * (x[i+1] - x[i]**2) + 2 * (x[i] - 1) + 200 * (x[i] - x[i-1]**2)
    grad[m-1] = 200 * (x[m-1] - x[m-2]**2)
    return grad

np.seterr(all='ignore')
n = 20
solution = np.ones(n)
x0 = 13*np.ones(n)
A = np.diag(np.arange(1,n+1))
b = np.arange(1,n+1)
plt.figure(figsize=(12, 12))
megiter = 10000
# test
def q_line_search(f, grad_f, x, d, alpha_init=10.0, rho=0.5, c1=1e-4, c2=0.9):
    alpha = np.dot(grad_f(x), grad_f(x))/(d @ A @ d)
    return alpha

i = 0
for beta in [PRP, HS, DY, N, MAD, SD, FR]: #CD, LS, PR
    i+= 1
    x_final, errors, slope = NLCG(rosenbrock, rosenbrock_grad, x0, beta, line_search, max_iter=megiter, solution=solution, mode=0)
    plt.semilogy(errors, label=beta.__name__, linewidth = 5, linestyle="solid" if i < 4 else "dotted")
    print(f"{beta.__name__:<10} final error: {np.round(np.linalg.norm(x_final-solution), 3):<10} degree: {np.round(-slope[0], 3):<5}")

    
plt.xlabel('Iteration')
plt.ylabel('f-error (log scale)')
plt.title(f'f-error vs Iteration')
plt.legend()
plt.ylim([1e-15, 100])
plt.grid(True)
plt.show()