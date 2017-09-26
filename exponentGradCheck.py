'''This function checks whether the (trickier) derivative wrt dGamma/dW (which involves taking a derivative 
inside the exponential function) is correct. Using method: http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization'''

import copy
import numpy as np 
from dGammadW import dGammadW
from estimateSignal import estimateSignal

def exponentGradCheck(sampleY,sampleX,alpha,W,stepSize,iteratorMu,C,sigma):
    
    # Initialize variables and arrays
    N=len(sampleY)
    iteratorN=np.arange(N)
    
    # My estimated dGamma / dW
    myExpDeriv=dGammadW(sampleY,alpha,W,stepSize,iteratorMu,sigma)
    
    
    # The correct dGamma /dW 
    expDerivCheck=np.zeros((N,N,N))
    expDerivCheck=np.asarray([np.asarray([speedExponentGradCheck(l,k,W,alpha,sampleX,sampleY,stepSize,iteratorMu,C,sigma) for l in iteratorN]) for k in iteratorN])
    expDerivCheck=expDerivCheck.swapaxes(0,2)
    
    
    # Calculate the difference
    expGradDiff=abs(myExpDeriv-expDerivCheck)
    
    
    # Return the results
    if (expGradDiff<.01).all():
        return ("RBF Grad wrt W is Correct")
    else:    
        return "RBF Grad wrt W is incorrect, max difference is: "+(str(np.max(expGradDiff)))

# Computes each entry of dGamma /dW 
def speedExponentGradCheck(l,k,W,alpha,sampleX,sampleY,stepSize,iteratorMu,C,sigma):
   
    # Estimates derivative using limit definition of derivative 
    epsilon=.000001
    WEpsPos=copy.deepcopy(W)
    WEpsPos[l][k]=WEpsPos[l][k]+epsilon
    WEpsNeg=copy.deepcopy(W)
    WEpsNeg[l][k]=WEpsNeg[l][k]-epsilon
    
    gammaEpsPos=sampleX-estimateSignal(WEpsPos,sampleY,alpha,C,stepSize,iteratorMu,sigma)
    gammaEpsNeg=sampleX-estimateSignal(WEpsNeg,sampleY,alpha,C,stepSize,iteratorMu,sigma)
    

    
    derivCheckEntry=(gammaEpsPos-gammaEpsNeg)/(2*epsilon)     
    
    return derivCheckEntry

