'''
% The purpose of this algorithm is to solve the interior subproblem for SCFA: 
% min(P) f(P) = tr(Pt*Atilde*P) + 2tr(Pt*B) + lambda ||Q*P||_21
% where Atilde = Qt*A*Q and Btilde = Qt*B.  
% Q is an n x q (k <= q <= 3k) orthonormal matrix whose 
% first k columns is the most recently computed W from SCFA 
% A is an n x n PSD matrix (therefore Atilde is PSD and q x q) lambda > 0.
% B is an n x k matrix (Btilde is then q x k).

% Input
% Atilde:  PSD matrix of size q x q
% Btilde:  q x k matrix 
% Q:       n x k orthonormal matrix used to generate Atilde and Btilde.
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.P = initial P of size q x k and the first k rows of P are the
    %          identity matrix and the rest is zeros.
    % opts.lambda = lambda > 0 used in f(P).  


% Output:  most recently computed P

% KKT Errors
%            ||AtildeP+Btilde + lambda*DtildeP-P*Lambda||_F/(||Atilde||_F + ||Btilde||_F + lambda*q)
%            where Lambda = PtAtildeP + Btilde + lambda PtDtildeP.  
%            and the second measuring ||PtBtilde - BtildetP||_F/(||Btilde||_F)
'''

import numpy as np
from numpy.linalg import norm

def scf_for_scfa_l21(Atilde,Btilde,Q,opts):
    
    # Initializing constants
    P = opts.P; lambda_param = opts.lambda_param
    q,k = Btilde.shape; n = Q.shape[0]
    
    # Initializing D (and equivalently Dtilde)
    D = np.zeros((n,1)); QP = Q@P
    for i in range(n):
        D[i] = 0.5/norm(QP[i,:])
        
    DQ = D*Q; Dtilde = Q.T@DQ
    
    nrmB = norm(Btilde,ord = 'fro')
    nrm_kkt = norm(Atilde,ord = 'fro') + nrmB + lambda_param*q
    
    # Calculate gradient G and lagrange multiplier matrix L
    AtildeP = Atilde@P; PAtildeP = P.T@AtildeP; PtBtilde = P.T@Btilde 
    DtildeP = Dtilde@P; PDtildeP = P.T@DtildeP 
    
    G= AtildeP + Btilde + lambda_param*DtildeP
    L = PAtildeP + PtBtilde + lambda_param*PDtildeP
    
    # Calculate resiudla matrix R
    R = G - P@L
    
    # Calculate initial KKT errors and objective value
    res_kkt = norm(R,ord = 'fro')/nrm_kkt
    res_sym = norm(PtBtilde-PtBtilde.T,ord = 'fro')/nrmB
    res_err = res_kkt + res_sym
    
    iter = 0
    
    while (opts.tol < res_err) and (opts.maxit > iter):
        
        # Construct J(P) to solve NEPv:  J(P)P = P*Psy
        # where Psy = P^T * J(P) and P is an orthonormal eigenbasis matrix 
        # of J(P) associated with its k smallest eigenvalues.
        
        PBtildeT = P@Btilde.T 
        DPPt = Dtilde@(P@P.T)
        J = Atilde + (PBtildeT + PBtildeT.T) + lambda_param*(DPPt + DPPt.T)
        J = (J+J.T)/2 
        
        Evals,U = np.linalg.eig(J)
        idx = np.argsort(Evals)
        P = U[:,idx[:k]]
        
        # Refinement step
        PtBtilde = P.T@Btilde 
        U,S,VT = np.linalg.svd(PtBtilde)
        UVt = -U@VT
        P = P@UVt
        
        # Recreate matrices
        PtBtilde = UVt.T@PtBtilde
        AtildeP = Atilde@P
        PAtildeP = P.T@AtildeP
        QP = Q@P
        
        # Recalculate matrices
        D = np.zeros((n,1))
        for i in range(n):
            D[i] = 0.5 / np.linalg.norm(QP[i, :])
        # Multiply each row of Q by the corresponding D[i] (broadcasting over columns).
        DQ = D * Q
        Dtilde = Q.T @ DQ
        DtildeP = Dtilde @ P
        PDtildeP = P.T @ DtildeP
        
        # Recalculate G andL
        G = AtildeP + Btilde + lambda_param * DtildeP
        L = PAtildeP + PtBtilde + lambda_param * PDtildeP
        
        # Recalculate residual matrix R
        R = G - P @ L
        
        # Calculate new errors
        res_kkt = np.linalg.norm(R, 'fro') / nrm_kkt
        res_sym = np.linalg.norm(PtBtilde - PtBtilde.T, 'fro') / nrmB
        res_err = res_kkt + res_sym
        
        # Update current iteration
        iter += 1
            
    return P
