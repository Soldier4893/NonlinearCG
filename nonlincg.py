import math
from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def get_objective_function(n, index=0):
    f, g, solution = None, None, None
    a = np.random.rand(n,n)
    A = a.T @ a + np.eye(n)
    if index == 0: # linear equation for SPD matrix
        solution = np.ones(n)
        b = A @ solution
        f = lambda x: 1/2*x@A@x-b@x
        g = lambda x: A@x-b
    elif index == 1: # tikhinov regularization
        C = np.random.rand(n,n)
        b = np.ones(n)
        solution = np.linalg.pinv(A@A + C@C) @ A.T@b
        f = lambda x: (A@x-b).T@(A@x-b) + (C@x).T @ (C@x)
        g = lambda x: 2*A.T @ (A @ x - b) + 2 * C.T @ C @ x
    elif index == 2: # noisy quadratic
        solution = np.ones(n)
        b = A @ solution
        f = lambda x: 1/2*x@A@x-b@x + np.random.normal(0, 0.01)
        g = lambda x: A@x-b+ np.random.normal(0, 0.01, n)
    else:
        raise ValueError('objective not recognized')
    return f, g, solution

# Use Armijo line search to satify wolfe conditions
def search(f, g, x, d, alpha0=10.0, rho=0.5, c1=1e-4, c2=0.9):
    alpha = alpha0
    grad = g(x)
    
    for _ in range(800):
        if f(x + alpha * d) <= f(x) + c1 * alpha * np.dot(grad, d) and\
            np.dot(g(x + alpha * d), d) >= c2 * np.dot(grad, d):
                return alpha
        alpha *= rho
    return 0

# exact line search for quadratic case
def q_search(f, g, x, d, A, alpha0=None, rho=None, c1=None, c2=None):
    r = g(x)
    return (r @ r) / (d @ A @ d)
# Nonlinear conjugate gradient
def NLCG(f, g, x0, compute_beta=None,  max_iter=1000, solution=None, mode = 'function'):
    def error_func(mode):
        if mode == 'f':
            return np.linalg.norm(f(x) - f(solution))
        elif mode == 'g':
            return np.linalg.norm(g(x))
        elif mode == 'x':
            return np.linalg.norm(x - solution)
        else:
            raise ValueError('errormode not recognized')
    
    # setup
    tol = {'f': 1e-11, 'g': 1e-11, 'x': 1e-11}[mode]
    errors = tol * np.ones(max_iter)
    x = x0
    r = g(x) #residual
    d = -r
    counter = max_iter
    
    # main loop
    for k in range(max_iter):
        errors[k] = error_func(mode)
        # stop algorithm if tolerance is met, to calculate convergence rate
        if error_func(mode) <= tol:
            counter = k
            errors = errors[:counter]
            break
        alpha = search(f, g, x, d)
        
        x_new = x + alpha * d
        g_new = g(x_new)
        
        # if k % len(x) == 0:
        #     beta = 0
        # else:
        beta = compute_beta(f, r, g_new, d, x, x_new, k)
        if np.isnan(beta):
            counter = k
            errors = errors[:counter]
            break
        d_new = -g_new + beta * d
        
        

        x = x_new
        r = g_new
        d = d_new

    return x, errors, calculate_rate(errors, counter)

# sometimes polyfit hits error
def calculate_rate(errors, M):
    relevant = errors[:M]
    y = np.log(relevant[1:])
    x = np.log(relevant[:-1])
    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        return np.zeros(2), None
    try:
        coeff = np.polyfit(x,y, 1)
    except:
        return np.zeros(2), None
    coeff[1] = np.e**coeff[1]
    q = coeff[0]
    c = coeff[1]
    

    # sublinear convergence, estimate p for convergence rate C/k^p
    if q < 0.97 or c >= 1:
        y1 = np.log(relevant)
        x1 = np.log(np.arange(1,M+1))
        coeff = np.polyfit(-x1, y1, 1)
        return coeff, False
    return coeff, True

def ag_search(f, g, y, alpha0=10.0, rho=0.5):
    grad = g(y)
    alpha = alpha0
    for _ in range(100):
        if f(y - alpha * grad) <= f(y) - alpha*np.dot(grad, grad)/2:
            return alpha
        alpha *= rho
    # line search failed
    return 0

def AG(f, g, x0, max_iter=1000, solution=None, mode='function'):
    def error_func(mode):
        if mode == 'f':
            return np.linalg.norm(f(x) - f(solution))
        elif mode == 'g':
            return np.linalg.norm(g(x))
        elif mode == 'x':
            return np.linalg.norm(x - solution)
        else:
            raise ValueError('errormode not recognized')
    
    # setup
    tol = {'f': 1e-11, 'g': 1e-8, 'x': 1e-8}[mode]
    errors = tol * np.ones(max_iter)
    x = x0
    y = x0
    counter = max_iter
    
    # main loop
    for k in range(max_iter):
        if error_func(mode) < tol:
            counter = k
            break
        alpha = ag_search(f, g, y)

        grad = g(y)
        x_new = y - alpha * grad
        y_new = x_new + k/(k+3)*(x_new - x)        
        errors[k] = error_func(mode)

        x = x_new
        y = y_new
    
    return x, errors, calculate_rate(errors, counter)

def FR(f, g, g_new, d, x, x_new, k):
    return np.linalg.norm(g_new)**2/np.linalg.norm(g)**2
def PRP(f, g, g_new, d, x, x_new, k):
    return np.dot(g_new, g_new-g) / np.linalg.norm(g)**2
def HS(f, g, g_new, d, x, x_new, k):
    if k % len(x) == 0:
        return 0
    return np.dot(g_new, g_new - g)/np.dot(d, g_new - g)
def DY(f, g, g_new, d, x, x_new, k):
    return np.dot(g_new, g_new)/np.dot(d, g_new-g)
def N(f, g, g_new, d, x, x_new, k):
    a = (g_new - g) - 2*d*np.dot(g_new-g, g_new-g)/np.dot(d, g_new-g)
    b = g_new / np.dot(d, g_new - g)
    return np.dot(a,b)
def LS(f, g, g_new, d, x, x_new, k):
    return -np.dot(g_new, g_new-g)/np.dot(d, g)
def CD(f, g, g_new, d, x, x_new, k):
    return -np.dot(g_new, g_new)/np.dot(d,g)
def MAD(f, g, g_new, d, x, x_new, k):
    n = len(x)
    mu = 0.99
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
    return l * beta
def SD(f, g, g_new, d, x, x_new, k):
    return 0

def rosenbrock(x):
    m = len(x)
    sum = 0
    for i in range(m-1):
        sum += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return sum
def rosenbrock_grad(x):
    m = len(x)
    grad = np.zeros(m)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1)
    for i in range(1, m-1):
        grad[i] = -400 * x[i] * (x[i+1] - x[i]**2) + 2 * (x[i] - 1) + 200 * (x[i] - x[i-1]**2)
    grad[m-1] = 200 * (x[m-1] - x[m-2]**2)
    return grad