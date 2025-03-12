'''
% This script solves the FSOR-l21 problem via SCFA-l21
% Find min(W) tr(W^T A W) + 2tr(W^TB) + ||W||_21
% where W lives on the Stiefel Manifold (n x k)

% A = X*X' and B = -X*Y'

% Input variables
% X - mean centered (or normalized) data matrix of size n x m (n features, m samples)
% Y - mean centered (or normalized) label matrix of size k x m (k classes, m samples)
% opts - options for SCFA-l21
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k (with orthonormal columns, and assuming opts.init = 1)
    % opts.lambda = lambda > 0 used in f(W).
    
'''

import numpy as np 
# from scipy.linalg import orth
from numpy.linalg import norm
import time
import scfa_l21
# import matplotlib.pyplot as plt

# import pandas as pd

class optimize_options:
    def __init__(self):
        self.tol = None
        self.maxit = None
        self.init = None
        self.lambda_param = None

class fsor_output_info:
    def __init__(self):
        self.time = None
        self.f = None
        self.W = None
        self.wts = None
        self.res_all = None

def fsor_l21(X,Y,opts):
    
    ctime = time.time()
    
    A = X@X.T; B = -X@Y.T
    scfa_info = scfa_l21.scfa_l21(A,B,opts)
    
    n = A.shape[0]
    wts = np.zeros((n,1))
    
    W = scfa_info.W
    for i in range(n):
        wts[i] = norm(W[i,:]) 
    wts = wts/sum(wts)
    
    ctime = time.time() - ctime
    info = fsor_output_info()
    info.time = ctime; info.f = scfa_info.f
    info.res_all = scfa_info.res_all
    info.W = W
    info.wts = wts
    
    return info

# if __name__ == "__main__":
    
#     m = 10000; n = 5000; k = 100;

#     X = np.random.rand(n,m); Y = np.random.rand(k,m)
#     X = (X - np.mean(X))/np.std(X)
#     Y = (Y - np.mean(Y))/np.std(Y)
#     W0 = orth(np.random.rand(n,k))

#     opts = optimize_options()
#     opts.tol = 1e-6; opts.maxit = 1000; 
#     opts.lambda_param = 1; opts.init = 1
#     opts.W = W0
    
#     info = fsor_l21(X,Y,opts)
#     plt.plot(info.f)
#     plt.show()
    
