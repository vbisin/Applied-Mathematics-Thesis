
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:36:58 2017

@author: vittoriobisin
"""
#from numba import jit,f8,int8

#@jit(f8[:,:](f8[:],int8))
def rbfF(signal,alphaDim):
    import numpy as np
    #import matplotlib.pyplot as plt
    # Where signal is a vector
    
    N=len(signal)
    N=N/1.
    gaussiansMatrix=np.zeros((int(N),alphaDim))
    sigma=30.
    #mu=N/alphaDim
    iteratorMu=np.zeros(alphaDim)
    iteratorMakeMu=np.arange(alphaDim/2)
    iteratorMakeMuRev=iteratorMakeMu[::-1]*(-30)-30

    iteratorMu[iteratorMakeMu]=iteratorMakeMuRev[iteratorMakeMu]
    iteratorMu[iteratorMakeMu+alphaDim/2+1]=iteratorMakeMu*30+30
    
    
    for i in range(int(N)):
        gaussiansMatrix[i,:]=np.asarray([np.exp((-(signal[i]-(center))**2.)/(2.*sigma**2.)) for center in iteratorMu]).astype(np.float64)
        

            
            
    #plt.plot(np.arange(len(signal))+1,signal)    
    #plt.plot(np.arange(len(signal))+1,np.sum(gaussiansMatrix,axis=1))
            
            
    return gaussiansMatrix
