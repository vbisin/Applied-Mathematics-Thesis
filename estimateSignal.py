'''The function evaluates the predicted signal given the current parameter inputs
(i.e. the RHS of the objective function)'''
import numpy as np
from rbf import rbfF

def estimateSignal(W,sampleY,alpha,C,stepSize,iteratorMu,sigma):

    #Estimate predicted signal (see paper for model description)
    estimatedSignal=np.dot(C,np.dot(rbfF(W,sampleY,len(alpha),stepSize,iteratorMu,sigma),alpha))
    
    # Returns predicted signal
    return estimatedSignal
