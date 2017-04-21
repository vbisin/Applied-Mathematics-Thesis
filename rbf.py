## Calculates Radial Basis Function matrix (of W*y), before multiplying by alpha 
# The Gaussians' centers are equally spaced, with a sigma equal to the interval between centers  

import numpy as np
def rbfF(W,sampleY,alphaDim):

    signal=np.dot(W,sampleY)
    N=len(signal)
    iteratorN=np.arange(N)
    gaussiansMatrix=np.zeros((N,alphaDim))
    
    # Compute step size and iterator for RBF 
    (stepSize,iteratorMu)=GRBFCenters(np.dot(W,sampleY),alphaDim)

    
    # Set sigma to be the length between Gaussian centers
    sigma=float(stepSize)
              
## Calculate matrix of RBF Gaussians for W*y       
    gaussiansMatrix=np.asarray([np.asarray([np.exp((-(signal[i]-center)**2.)/(2.*sigma**2.)) for center in iteratorMu]).astype(np.float64) for i in iteratorN])
            
    return gaussiansMatrix


def GRBFCenters(signal,alphaDim):
    # Define min and max of the input signal
    minSig=min(signal)
    maxSig=max(signal)
    
    # Calculate total length of the interval between the max and min  
    if maxSig<0 and minSig>0:
        print("code needs to be fixed here")
    interval=abs(int(maxSig)-int(minSig))
    
    # Keep increasing interval until it is divisible by the dimension of alpha
    freedom=False
    while freedom==False:
        if interval%(alphaDim-1)!=0:
            interval=interval+1
        else:
            break
    
    # Calculate step size so that given current dimension of alpha, we can span the entire interval
    stepSize=interval/float(alphaDim-1)
    
    # Make array with corresponding step size so that spans entire interval
    iteratorMakeMu=np.arange(alphaDim)*stepSize
    
    # Define an array using step size so that it covers the entire interval                         
    iteratorAlpha=np.arange(alphaDim)
    iteratorMu=np.zeros(alphaDim)
    minSig=int(min(signal))
    iteratorMu[iteratorAlpha]=iteratorMakeMu[iteratorAlpha]+minSig   
        
    return (stepSize, iteratorMu)