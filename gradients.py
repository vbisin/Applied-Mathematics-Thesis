## Solves for gradients of loss function - w.r.t alpha and W


import numpy as np
from rbf import rbfF
from dOmegadW import dOmegadW
from estimateSignal import estimateSignal


## Computes the dFdGamma derivative used in both the alpha and W gradients
def dFdGamma(sampleX,sampleY,alpha,W):
    
    ## The dFdGamma is equivalent to the current estimate of the loss function
    dFdG=sampleX-(estimateSignal(W,sampleY,alpha))
    
    return dFdG


## Compute the gradient w.r.t. alpha
def alphaGradient(sampleX,sampleY,alpha,W):
    
    # Recover respective lengths
    N=len(sampleX)    
    alphaDim=len(alpha)
    
    # Define arrays for faster computations
    iteratorN=np.arange(N)
    iteratorAlpha=np.arange(alphaDim)
    
    # Define variables needed for calculating derivative
    dGdA=np.zeros((N,alphaDim))
    rbfWY=rbfF(W,sampleY,alphaDim)
    negWTrans=-np.transpose(W)
    
## Calculate dGammaddAlpha 
    dGdA=np.asarray([np.asarray([np.dot(negWTrans[i,:],(rbfWY[:,j])) for j in iteratorAlpha]).astype(np.float) for i in iteratorN])
    
    
## Calculate dot product from chain rule (dFdGamma=dFdGamma*dGammadAlpha) 
    dFdG=dFdGamma(sampleX,sampleY,alpha,W)
    
    dFdA=np.dot(dFdG,dGdA) 
    
    return dFdA






## Compute the derivative w.r.t. W
def WGradient(sampleX,sampleY,alpha,W):
    
    # Recover respective lengths
    N=len(sampleX)
    alphaDim=len(alpha)
    negTransW=-np.transpose(W)
    
    
    # Define omega as the Gaussian RBF Function
    rbfWY=rbfF(W,sampleY,alphaDim)
    omega=np.dot(rbfWY,alpha)

## Calculate LHS of product rule for dGammadW (i.e. -transpose(W)*dOmegadW)
    derivOmegadW=dOmegadW(sampleY,alpha,W)
    LHSProductRule=np.dot(negTransW,derivOmegadW)

    
## Calculate RHS of product rule for dGammadW (i.e. d(-transpose(W))dW*omega) 
    #since d(-transpose(W))dW is the identity 4-tensor, then the above product will be a 3 tensor
    RHSProductRule=np.zeros((N,N,N))
    iterator=np.arange(N)
    #LHSProductRule[iterator,iterator,iterator]=-omega[iterator]
    RHSProductRule[iterator,iterator,iterator]=-omega[iterator]


## Calculate dFdW (i.e. dFdG*dGammadW)
    dFdG=dFdGamma(sampleX,sampleY,alpha,W)

    dFdW=np.dot(dFdG,LHSProductRule+RHSProductRule)

    return dFdW
