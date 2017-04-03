## Returns dOmegadW derivative 
# Where omega is defined as the RBF function

import numpy as np

def dOmegadW(sampleY,alpha,W):
    
    #Recover length of signal  
    N=len(sampleY)
    
    #Variables needed for derivative calculation
    iterator=np.arange(N)
    dOmegadW=np.zeros((N,N,N))    
    
##Calculate dOmegadW     
    for i in iterator:
        lvector=np.array([dOmegadWEntry(sampleY,l,W,i,alpha) for l in iterator]).astype(np.float)    
        dOmegadW[i,i,:]=lvector
            
    return dOmegadW


def dOmegadWEntry(sampleY,l,W,i,alpha):

## Set up parameters    
    alphaDim=len(alpha)
    signal=np.dot(W[i,:],sampleY)
    
    # Set sigma (standard deviation of each Gaussian) as the interval between each Gaussian center
    sigma=30.


## Define centers of Gaussian RBF, so that centers of Gaussians match interval 
# of ranges of the signals  
    iteratorMu=np.zeros(alphaDim)
    iteratorMakeMu=np.arange(alphaDim/2)    

    # First define iterator array with values in the negative image range of the signal 
    iteratorMakeMuRev=iteratorMakeMu[::-1]*(-30)-30
    iteratorMu[iteratorMakeMu]=iteratorMakeMuRev[iteratorMakeMu]
    
    # Define iterator array with values in the positive image range of the signal 
    iteratorMu[iteratorMakeMu+alphaDim/2+1]=iteratorMakeMu*30+30
    

    matrixEntry=np.zeros(alphaDim)
    iterator=np.arange(alphaDim)        

## Compute respective entry (i.e. the i,i,l) of dOmegadW 
    matrixEntry[iterator]=np.exp(-(signal-iteratorMu[iterator])**2./(2.*sigma**2.))*(-sampleY[l]*alpha[iterator]*(signal-iteratorMu[iterator])/(sigma**2.))
    matrixEntry=sum(matrixEntry)
    return matrixEntry



