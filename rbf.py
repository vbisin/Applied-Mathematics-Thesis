## Calculates Radial Basis Function matrix (of W*y), before multiplying by alpha 
# The Gaussians' centers are equally spaced, with a sigma equal to the interval between centers  

import numpy as np
def rbfF(signal,alphaDim):

    
    N=len(signal)
    
    gaussiansMatrix=np.zeros((int(N),alphaDim))
    
    # Set sigma to be the length between Gaussian centers
    sigma=30.
    
    # First define iterator array with values in the negative image range of the signal 
    iteratorMu=np.zeros(alphaDim)
    iteratorMakeMu=np.arange(alphaDim/2)
    iteratorMakeMuRev=iteratorMakeMu[::-1]*(-30)-30

    # Define iterator array with values in the positive image range of the signal 
    iteratorMu[iteratorMakeMu]=iteratorMakeMuRev[iteratorMakeMu]
    iteratorMu[iteratorMakeMu+alphaDim/2+1]=iteratorMakeMu*30+30
    
## Calculate matrix of RBF Gaussians for W*y       
    for i in range(int(N)):
        gaussiansMatrix[i,:]=np.asarray([np.exp((-(signal[i]-(center))**2.)/(2.*sigma**2.)) for center in iteratorMu]).astype(np.float64)
        
            
    return gaussiansMatrix
