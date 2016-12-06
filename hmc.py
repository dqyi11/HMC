'''
Created on Dec 6, 2016

@author: daqingy
'''

import numpy as np

def LeapFrog(X0, P0, delta, L, dU, dK):
    # simulate Hamiltonian dynamics
    # first 1/2 step of momentum
    P_star = P0 - delta / 2 * dU( X0 )   
    
    # first full step for position/sample
    X_star = X0 + delta * dK( P_star )
    
    # full steps
    for jL in range(L-1):
        # momentum
        P_star = P_star - delta * dU( X_star )
        # position / sample
        X_star = X_star + delta * dK( P_star )
        
    # last half step
    P_star = P_star - delta / 2 * dU(X_star)
    
    return X_star, P_star

def HMC(sample_num, dim, X0, delta, L, U, dU, K, dK):

    # initial state
    X = np.zeros((dim, sample_num))
    X = np.matrix(X)

    X[:,0] = X0
    
    t = 0
    while t < sample_num-1:
        t = t+1
        
        # sample random momentum
        P0 = np.random.randn(dim,1)
        P0 = np.matrix(P0)
        
        
        X_star, P_star = LeapFrog(X[:,t-1], P0, delta, L, dU, dK)
        
        # evaluate energy 
        U0 = U(X[:,t-1])
        U_star = U(X_star)
        
        K0 = K(P0)
        K_star = K(P_star)
        
        # acceptance/rejection criterion
        alpha = np.min([1, np.exp((U0+K0)-(U_star+K_star))])
        
        u = np.random.rand()
        if u <= alpha:
            X[:,t] = X_star
        else:
            X[:,t] = X[:,t-1]     
    
    return X