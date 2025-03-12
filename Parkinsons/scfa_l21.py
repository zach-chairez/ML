'''
% The purpose of this algorithm is to solve: 
% min(W) f(W) = tr(WtAW) + 2tr(WtB) + lambda ||W||_21
% where A(nxn) is an PSD matrix, B(nxk) is arbitrary
% W(nxk) (we assume that k <= n) with orthonormal columns
% and lambda > 0.  

% Input
% A:  PSD matrix of size n x n
% B:  n x k matrix 
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k(with orthonormal columns, and assuming opts.init = 1)
    % opts.lambda = lambda > 0 used in f(W).  


% info:
% info.f = 1 x ___ array of objective values f(W) over each iteration
% info.res = 2 x ____ array of normalized residual errors with the 
%            first entry measuring ||AW+B + lambda*DW-W*Lambda||_F/(||A||_F + ||B||_F + lambda*n)
%            where Lambda = WtAW + WtB + lambda WtDW.  
%            and the second measuring ||WtB - BtW||_F/(||B||_F)
% info.time = cputime (secs) from start to finish of algorithm
% info.W = most recently W computed 
'''

import time
import numpy as np
from numpy.linalg import norm
from scipy.linalg import orth
import scf_for_scfa_l21

class optimize_options:
    def __init__(self):
        self.tol = None
        self.maxit = None
        self.P = None
        self.W = None
        self.lambda_param = None

class scfa_output_info:
    def __init__(self):
        self.time = None
        self.f = None
        self.res_all = None
        self.W = None
        
def scfa_l21(A,B,opts):

    # Initialize constants
    total_time = time.time()
    n,k = B.shape
    lambda_param = opts.lambda_param
    two_lambda = 2*lambda_param
    
    # For interior SCF function
    opts_in = optimize_options()
    opts_in.maxit = np.ceil(opts.maxit/3)
    opts_in.lambda_param = opts.lambda_param
    
    # Initializing diagonal vector vecD
    vecD = np.zeros((n,1))
    
    # Initilizing W
    if opts.init == 1:
        W = opts.W 
    elif opts.init == 0:
        W = orth(np.random.rand(n,k))
    else:
        print("You suck, enter 0 or 1 next time")
        return 

    # Constructing matrices for objective function and KKT errors
    nrmA = norm(A,ord='fro'); nrmB = norm(B,ord='fro')
    nrm_kkt = nrmA + nrmB + opts.lambda_param*n
    
    AW = A@W; WAW = W.T@AW; 
    for i in range(n):
        vecD[i] = 0.5/norm(W[i,:])
    
    DW = W*vecD; WDW = W.T@DW
    WtB = W.T@B
    
    # Gradient G and lagrange multiplier matrix L
    G = AW + B + lambda_param*DW
    L = WAW + WtB + lambda_param*WDW
    
    # Residual matrix R (note that R is orthgonal to W)
    R = G - W@L
    
    # Calculating current objective function value
    f = np.trace(WAW) + 2*np.trace(WtB) + two_lambda*np.trace(WDW)
    f_all = [f]
    
    # Calculate current individual KKT errors
    res_kkt = norm(R,ord='fro')/nrm_kkt
    res_sym = norm(WtB-WtB.T,ord='fro')/nrmB
    
    # Calculate sum of KKT errors and save for output
    res_err = res_kkt + res_sym
    res_all = [[res_kkt,res_sym]]
    
    error_check = True 
    iter = 0; Wp = W
    
    while error_check and (opts.maxit > iter):
        
        # Find a matrix Q such that Q is an orthonormal basis
        # matrix of [W R Wp] and the first k columns of 
        # Q is W
        
        if iter >= 1:
            R = np.hstack((R,Wp-W@(W.T@Wp)))
            
        R = orth(R); R = R - W@(W.T@R)
        R = orth(R); Q = np.hstack((W,R))
    
        # Projecting onto the subspace spanned by the columns of Q = [W R]
        AR = A@R; AQ = np.hstack((AW,AR)); WAR = W.T@AR
        Atilde = np.block([[WAW,WAR],[WAR.T,R.T@AR]])
        Btilde = np.vstack((WtB,R.T@B))
        
        # Forces next approximation to be slightly better
        opts_in.tol = max(0.5*opts.tol,0.25*res_err)
        
        # Solve internal subproblem
        q,k = Btilde.shape
        opts_in.P = np.eye(q,k)
        P = scf_for_scfa_l21.scf_for_scfa_l21(Atilde,Btilde,Q,opts_in)
        
        # Update matrices for next iteration
        Wp = W.copy()
        W = Q@P 
        AW = AQ@P
        
        # Calculate current KKT error and objective value
        WAW = W.T@AW
        for i in range(n):
            vecD[i] = 0.5/norm(W[i,:])
        DW = vecD*W; WDW = W.T@DW
        WtB = W.T@B
        
        G = AW + B + lambda_param*DW
        L = WAW + WtB + lambda_param*WDW
        
        f = np.trace(WAW) + 2*np.trace(WtB) + two_lambda*np.trace(WDW)
        f_all.append(f)
        
        f_error = abs((f_all[iter]-f)/f)
        
        res_kkt = norm(R,ord = 'fro')/nrm_kkt
        res_sym = norm(WtB-WtB.T,ord = 'fro')/nrmB
        res_err = res_kkt + res_sym
        
        error_check = (res_err > opts.tol) and (f_error > opts.tol*0.1)
        
        res_all.append([res_kkt,res_sym])
        
        iter += 1
    
    total_time = time.time() - total_time
    info = scfa_output_info()
    info.time = total_time; info.W = W
    info.res_all = res_all 
    info.f = f_all
    
    return info
