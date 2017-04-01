## Returns dOmegadW derivative 
# Where omega is defined as the RBF vector

import numpy as np
from dOmegadWEntry import dOmegadWEntry

def makedOmegadW(sampleY,alpha,W):
    
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
    





