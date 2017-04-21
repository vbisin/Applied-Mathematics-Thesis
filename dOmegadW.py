 ## Returns dOmegadW derivative 
# Where omega is defined as the RBF function

import numpy as np
from rbf import GRBFCenters
def dOmegadW(sampleY,alpha,W):
    
    #Recover length of signal  
    N=len(sampleY)
    #Variables needed for derivative calculation
    iteratorN=np.arange(N)
    derivOmegadW=np.zeros((N,N,N))    
   
    
## Define centers of Gaussian RBF, so that centers of Gaussians match interval 
# of ranges of the signals  

    # Compute step size and iterator for RBF 
    (stepSize,iteratorMu)=GRBFCenters(np.dot(W,sampleY),len(alpha))


    # Set sigma (standard deviation of each Gaussian) as the interval between each Gaussian center
    sigma=float(stepSize)          
    
    
##Calculate dOmegadW    
    for i in range(N): 
        lvector=np.array([dOmegadWEntry(sampleY,l,W[i,:],alpha,sigma,iteratorMu) for l in iteratorN]).astype(np.float)    
        derivOmegadW[i,i,:]=lvector
            
    return derivOmegadW


def dOmegadWEntry(sampleY,l,Wvector,alpha,sigma,iteratorMu):

## Set up parameters    
    alphaDim=len(alpha)
    signal=np.dot(Wvector,sampleY)
    
    
    matrixEntry=np.zeros(alphaDim)
    iteratorAlpha=np.arange(alphaDim)        

## Compute respective entry (i.e. the [i,i,l] entry) of dOmegadW 
    matrixEntry[iteratorAlpha]=((-sampleY[l]*alpha[iteratorAlpha]*(signal-iteratorMu[iteratorAlpha]))/(sigma**2))*(np.exp((-(signal-iteratorMu[iteratorAlpha])**2.)/(2.*sigma**2.)))
    matrixEntry=sum(matrixEntry)
    
   
    return matrixEntry