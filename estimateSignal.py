## Evaluates RHS of loss function (i.e. returns the predicting signal)

import numpy as np
from rbf import rbfF

def estimateSignal(W,sampleY,alpha):

    
    signal=np.dot(rbfF(W,sampleY,len(alpha)),alpha) 
    signal=sampleY+np.dot(np.transpose(W),signal)
    
    # Returns predicted signal
    return signal
