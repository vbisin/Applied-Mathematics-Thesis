## Solves for gradients of loss function - w.r.t alpha and W


import numpy as np
from rbf import rbfF
from numpy import dot as dot
from numpy import transpose as trans
from makedOmegadW import makedOmegadW
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
    rbfWY=rbfF(dot(W,sampleY),alphaDim)
    negWTrans=-trans(W)
    
## Calculate dGammaddAlpha 
    for i in iteratorN:
        dGdA[i,:]=np.asarray([dot(negWTrans[i,:],(rbfWY[:,j])) for j in iteratorAlpha]).astype(np.float) 
    
    
## Calculate dot product from chain rule (dFdGamma=dFdGamma*dGammadAlpha) 
    dFdG=dFdGamma(sampleX,sampleY,alpha,W)
    dFdA=dot(dFdG,dGdA) 
    
    return dFdA






## Compute the derivative w.r.t. W
def wGradient(sampleX,sampleY,alpha,W):
    
    # Recover respective lengths
    N=len(sampleX)
    alphaDim=len(alpha)
    negTransW=-trans(W)
    
    
    # Define omega as the Gaussian RBF Function
    rbfWY=rbfF(dot(W,sampleY),alphaDim)
    omega=dot(rbfWY,alpha)
    
## Calculate LHS of product rule for dGammadW (i.e. d(-transpose(W))dW*omega) 
    #since d(-transpose(W))dW is the identity 4-tensor, then the above product will be a 3 tensor
    LHSProductRule=np.zeros((N,N,N))
    iterator=np.arange(N)
    LHSProductRule[iterator,iterator,iterator]=-omega[iterator]
    
    
## Calculate RHS of product rule for dGammadW (i.e. -transpose(W)*dOmegadW)
    dOmegadW=makedOmegadW(sampleY,alpha,W)
    RHSProductRule=dot(negTransW,dOmegadW)


## Calculate dFdW (i.e. dFdG*dGammadW)
    dFdG=dFdGamma(sampleX,sampleY,alpha,W)

    dFdW=dot(dFdG,trans(LHSProductRule+RHSProductRule))
    
    return dFdW