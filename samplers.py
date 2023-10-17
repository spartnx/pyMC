import numpy as np
import matplotlib.pyplot as plt

#%% Cutpoint method 
def cmset(cdf, m, params=()):
    I = []
    j, k, A = 0, -1, 0
    while j < m:
        while A <= j:
            k += 1
            A += m*cdf(k, *params)
        j += 1
        I.append(k)
    return I

def cm(cdf, m, n=1e6, seed=0, params=()):
    n = int(n)
    rng = np.random.Generator(np.random.MT19937(seed))
    I = cmset(cdf, m, params)
    U = rng.uniform(size=(n,))
    L = np.floor(m*U).astype(int)
    X = np.array([I[l] for l in L], dtype=int)
    index_search_fcn = lambda u,x: index_search(u, x, cdf, params)
    Y = np.vectorize(index_search_fcn)(U, X)
    return Y

def index_search(u, x, cdf, params):
    while u > cdf(x,*params):
        x += 1
    return x

def cdf_estimator(k, Y):
    X = Y - k
    X[X <= 0] = -1
    X[X > 0] = 0
    X = -X
    return sum(X)/len(X)

def pmf_estimator(k, Y):
    X = Y - k
    X[X!=0] = -1
    X[X==0] = 1
    X[X<0] = 0
    return sum(X)/len(X) 
