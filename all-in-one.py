import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def CG(x, A, b):
    r_k = A @ x - b
    p = -r_k
    beta = 0
    for _ in range(100):
    # while np.linalg.norm(r_k) > 1e-16:
        alpha = (r_k @ r_k) / (p @ A @ p)
        x = x + alpha * p
        r_k1 = r_k + alpha * A @ p
        beta = (r_k1 @ r_k1) / (r_k @ r_k)
        p = -r_k1 + beta * p
        r_k = r_k1
        if np.linalg.norm(r_k1) < 1e-16:
            return x
    return x

n=200
A = np.ones((n,n)) + np.diag(np.arange(n)) 
b = A @ np.arange(n)

def f(x):
    return 1/2 * (x @ A @ x) - (x @ b)

def g(x):
    return A @ x - b


def line_search_newt(f, g, x, p):
    alpha = 1

    for _ in range(20):
        alpha_1 = alpha - f(x+alpha*p)/(p @ g(x+alpha*p))
        alpha = alpha_1
    return alpha
def line_search_bin(f, g, x_k, p_k):
    alpha = 1

    for _ in range(100):
        # wc = wolfe_condition(x_k, p_k,f, g, alpha)
        if f(x_k + alpha*p_k) <f(x_k):
            return alpha
        else:
            alpha = alpha/2
    # r_k = g(x)
    # print((r_k @ r_k) / (r_k @ A @ r_k))
    # print(alpha)
    return alpha

def wolfe_condition(x_k, p_k,f, g, alpha):
    C1 = 0.0001
    C2 = 0.1
    f_k = f(x_k)
    f_k1 = f(x_k+alpha*p_k)
    g_k = g(x_k)
    g_k1 = g(x_k+alpha*p_k)
    return f_k1 <= f_k+C1*alpha*p_k @ g_k and -p_k@g_k1 <= -C2*p_k@g_k

def NLCG(x_k, f, g, method='FR'):
    r_k = g(x_k)
    p_k = -r_k
    
    beta_methods = {
        'FR': lambda g_k1, g_k, p_k: (g_k1 @ g_k1) / (g_k @ g_k),
        'PRP': lambda g_k1, g_k, p_k: (g_k1 @ (g_k1-g_k)) / (g_k @ g_k),
        'CD': lambda g_k1, g_k, p_k: (g_k1 @ g_k1) / (-p_k @ g_k),
        'LS': lambda g_k1, g_k, p_k: (g_k1 @ (g_k1-g_k)) / (-p_k @ g_k),
        'DY': lambda g_k1, g_k, p_k: (g_k1 @ g_k1) / (p_k @ g_k),
        'N': lambda g_k1, g_k, p_k: ((g_k1-g_k)-2*p_k*((g_k1-g_k) @ (g_k1-g_k) / p_k @ (g_k1-g_k))) @ (g_k1) / (p_k @ (g_k1-g_k))
    }

    for _ in range(n):
        # alpha = prinline_search(f, g, x_k, p_k)
        ls = line_search_bin(f, g, x_k, p_k)
        ls2 = line_search_newt(f, g, x_k, p_k)
        alpha = (g(x_k) @ g(x_k)) / (p_k @ A @p_k)
        alpha = ls

        # print('linesearch: ', line_search(f, g, x_k, p_k), 'alpha: ', alpha)
        x_k1 = x_k + alpha * p_k
        beta = beta_methods[method](g(x_k1), g(x_k), p_k)
        p_k1 = -g(x_k1)+beta * p_k
        p_k = p_k1
        x_k = x_k1
    return x_k

np.seterr(divide='ignore', invalid='ignore')

# print(np.linalg.norm(CG(x, A, b)- np.linalg.solve(A, b)))

x_0=np.zeros(n)
y1 = NLCG(x_0, f, g)
y2 = np.linalg.solve(A, b)
print(np.round(y1,4), y2,np.linalg.norm(y1-y2))
