'''
Created on Dec 6, 2016

@author: daqingy
'''

from hmc import HMC
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # step size
    delta = 0.3
    sample_num = 1000
    L = 20
    
    mu = np.zeros((2,1))
    var = np.matrix([[1, .8],[.8, 1]])

    # potential energy function
    def U_func(X):    
        U = X.T * np.linalg.inv(var) * X / 2
        return np.sum(U)
    
    # gradient potential energy function
    def dU_func(X):
        A = np.linalg.inv(var) * X
        return A
        
    # kinetic energy function    
    def K_func(P):
        s = np.sum( P.T * P / 2 )
        return s
    
    # gradient kinetic energy function
    def dK_func(P):
        return P
    
    X0 = np.matrix([[0],[6]])
    X = HMC(sample_num, 2, X0, delta, L, U_func, dU_func, K_func, dK_func)
    
    s_mu = np.mean(X,1)
    s_var = np.cov(X)
    
    print s_mu
    print s_var
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0,:],X[1,:])
    ax.plot(X[0,0:50],X[1,0:50],'r')
    ax.set_xlim([-6,6])
    ax.set_ylim([-6,6])
    
    plt.show()