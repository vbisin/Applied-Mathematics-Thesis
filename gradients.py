'''This function calculates the gradients of the objective function - w.r.t alpha and W ''' 

import numpy as np
from rbf import rbfF
from dGammadW import dGammadW
from estimateSignal import estimateSignal
import scipy 


## Computes the dFdGamma derivative used in both the alpha and W gradients (see gradients appendix in writeup)
def dFdGamma(sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma):
    

    dFdG=sampleX-estimateSignal(W,sampleY,alpha,C,stepSize,iteratorMu,sigma)

    return dFdG


## Computes the gradient w.r.t. alpha
def alphaGradient(alpha,sampleX,sampleY,W,C,stepSize,iteratorMu,sigma):
    
    # Recover respective lengths
    N=len(sampleX)    
    alphaDim=len(alpha)
    
    # Define arrays
    iteratorN=np.arange(N)
    iteratorAlpha=np.arange(alphaDim)
    
    # Define variables needed for calculating derivative
    dGdA=np.zeros((N,alphaDim))
    rbfWY=rbfF(W,sampleY,alphaDim,stepSize,iteratorMu,sigma)
    rbfWY=np.dot(C,rbfWY)

## Calculate dGammaddAlpha 
    dGdA=np.asarray([np.asarray([(-rbfWY[i,j]) for j in iteratorAlpha]).astype(np.float) for i in iteratorN])
    
## Calculate dot product from chain rule (dFdAlpha=dFdGamma*dGammadAlpha) 
    dFdG=dFdGamma(sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma)  
    dFdA=np.dot(dFdG,dGdA) 
    
    return dFdA


## Computes the derivative w.r.t. W
def WGradient(W,sampleX,sampleY,alpha,C,stepSize,iteratorMu,sigma):

## Calculate  dGammadW
    derivGammadW=dGammadW(sampleY,alpha,W,stepSize,iteratorMu,sigma)
    derivGammadW=-np.dot(C,derivGammadW)
                         
## Calculate dFdW (dFdW= dFdG*dGammadW)
    dFdG=dFdGamma(sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma)
    dFdW=scipy.tensordot(dFdG,derivGammadW, axes=[0,0])
        
    return dFdW



       
     
     
     
     
     
     
     
