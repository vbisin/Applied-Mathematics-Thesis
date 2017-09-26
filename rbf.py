''' Calculates the Radial Basis Function matrix (M basis function for each N signals => NxM matrix), before multiplying by alpha. '''

import numpy as np
def rbfF(W,sampleY,alphaDim,stepSize,iteratorMu,sigma):
    
    # Input signal for the RBF (the linearly transformed signal)
    convolvedSignal=np.convolve(sampleY,W,'same')

    # Recover variables and define arrays
    N=len(convolvedSignal)
    iteratorN=np.arange(N)
    iteratorAlpha=np.arange(alphaDim)

    # Calculate Gaussian RBF          
    gaussiansMatrix=np.asarray([np.asarray([np.exp((-(convolvedSignal[i]-iteratorMu[j])**2.)/(2.*sigma**2.)) for j in iteratorAlpha]).astype(np.float64) for i in iteratorN])

    return gaussiansMatrix
