'''Calculates the equally spaced centers of the RBF (and its dimension) given 
the stepSize parameter (i.e. the spacing between each one of the RBF centers) 
and the initial linearly transformed signal for each one of the samples'''

import numpy as np

def GRBFCenters(noisyDerivs,stepSize):

     # Calculate the minimum and maximum of all the linearly transformed signals
     minSig=int(np.min(noisyDerivs))-20
     maxSig=int(np.max(noisyDerivs))+20
     
     # Define the length between these two extrema          
     absMaxSig=max(abs(minSig),abs(maxSig)) 
     interval=2*absMaxSig
     
     # Calculate a dimension of alpha which covers this interval given the initial 
     # stepSize parameter
     alphaDim=interval/float(stepSize)
     if alphaDim%1!=1:
         alphaDim=int(alphaDim+.5)


     # Calculate the array (iteratorMu) with value the RBF centers
     iteratorAlpha=np.arange(alphaDim)
     
     iteratorMu=np.zeros(alphaDim)                      
     iteratorMakeMu=np.arange(alphaDim)*stepSize
     iteratorMu[iteratorAlpha]=iteratorMakeMu[iteratorAlpha]-(absMaxSig)  

     # Return the dimension of the RBF and its centers
     return (alphaDim,iteratorMu)