import scipy.linalg
import numpy as np
import pyiacsun as ps
import matplotlib.pyplot as pl

# pyZeroSR1
M = 200
N = 1000
K = 10
mu = 0.02
sigma = 0.01

# Create sparse signal
x = np.zeros(N)
ind = np.random.permutation(N)
x[ind[0:K]] = 1.0

# Define matrix
AMat = np.random.normal(size=(M,N))
AMat /= np.linalg.norm(AMat, 2)

# Define observation vector
b = AMat.dot(x)
# b += np.random.normal(scale=sigma, size=b.shape)

L = np.linalg.norm(AMat, 2)**2

A = lambda z : AMat.dot(z)
At = lambda z : AMat.T.dot(z)

prox = lambda x0, d, u, varargin = None : ps.sparse.proxes_rank1.prox_rank1_l1(x0, d, u, mu)
h = lambda x : mu * np.linalg.norm(x,1)
fcnGrad = lambda x : ps.sparse.smooth.normSquared(x, A, At, np.atleast_2d(b).T)

opts = {'tol': 1e-10, 'grad_tol' : 1e-6, 'nmax' : 1000, 'verbose' : False, 'N' : N, 'L': L, 'verbose': 25}

xk, nIteration, stepSizes = ps.sparse.zeroSR1(fcnGrad, h,prox, opts)


# FASTA
# Define the operators we need
f = lambda x : 0.5 * np.linalg.norm(x - b, 2)**2
gradf = lambda x : x - b
g = lambda x : mu * np.linalg.norm(x, 1)
proxg = lambda x, t: ps.sparse.proxes.prox_l1(x, t*mu)

values = np.zeros(N)
out = ps.sparse.fasta(A, At, f, gradf, g, proxg, values, verbose=True, tol=1e-12, maxIter=60000, accelerate=True, backtrack=False, adaptive=False)
values = out.optimize()

pl.plot(x)
pl.plot(values,'o')
pl.plot(xk, 'x')