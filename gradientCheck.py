'''This function checks whether the gradients wrt alpha and W are correct. 
Using method: http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization'''

import numpy as np
from estimateSignal import estimateSignal
from gradients import alphaGradient, WGradient
import copy

def alphaGradCheck(sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma):
    
    # Initialize variables and arrays
    alphaDim=len(alpha)
    alphaIterator=np.arange(alphaDim)


    #Compute my gradient wrt alpha
    myAlphaGrads=alphaGradient(alpha,sampleX,sampleY,W,C,stepSize,iteratorMu,sigma)
    
    
    # Compute correct gradient wrt alpha 
    correctAlphaGrads=np.asarray([speedAlphaGradCheck(i,sampleX,sampleY,W,alpha,C,stepSize,iteratorMu,sigma) for i in alphaIterator])
    
    
    # Returns the difference 
    if (abs(myAlphaGrads-correctAlphaGrads)<.001).all():
        return("Alpha Gradient is correct")
    else:
        return("Alpha Gradient is incorrect")
    
def speedAlphaGradCheck(i,sampleX,sampleY,W,alpha,C,stepSize,iteratorMu,sigma):
    
        # Estimates derivative using limit definition of derivative per entry
        epsilon=.0001
        alphaEpsPos=copy.deepcopy(alpha)
        alphaEpsPos[i]=alphaEpsPos[i]+epsilon
        alphaEpsNeg=copy.deepcopy(alpha)         
        alphaEpsNeg[i]=alphaEpsNeg[i]-epsilon
        
        fPos=sum(.5*((sampleX-estimateSignal(W,sampleY,alphaEpsPos,C,stepSize,iteratorMu,sigma))**2))
        fNeg=sum(.5*((sampleX-estimateSignal(W,sampleY,alphaEpsNeg,C,stepSize,iteratorMu,sigma))**2))
        
        derivativeCheck=(fPos-fNeg)/(2*epsilon)          
        
        return derivativeCheck
          
             
def WGradCheck(sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma):
    
    # Initialize variables and arrays
    WIterator=np.arange(len(W))
    
    # My calculated W derivative 
    myWGrads=WGradient(W,sampleX,sampleY,alpha,C,stepSize,iteratorMu,sigma)

    
    #Compute my gradient wrt W
    correctWGrads=np.asarray([speedWGradCheck(k,sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma)for k in WIterator])

    
    # Calculate matrix of differences 
    gradDifference=abs(correctWGrads-myWGrads)
    
    #Retrun the difference 
    if ((gradDifference)<.01).all():
        return ("W Gradient is correct")
    else:    
        return "W Gradient is incorrect, max difference is: " + (str(np.max(gradDifference)))   

 
## Sub function to speed up WGradCheck
def speedWGradCheck(k,sampleX,sampleY,alpha,W,C,stepSize,iteratorMu,sigma):    
    
    
    # Estimates derivative using limit definition of derivative per entry
    epsilon=.000001
    WEpsPos=copy.deepcopy(W)
    WEpsPos[k]=WEpsPos[k]+epsilon
    WEpsNeg=copy.deepcopy(W)
    WEpsNeg[k]=WEpsNeg[k]-epsilon
    
    
   
    fPos=sum(.5*((sampleX-estimateSignal(WEpsPos,sampleY,alpha,C,stepSize,iteratorMu,sigma))**2))
    fNeg=sum(.5*((sampleX-estimateSignal(WEpsNeg,sampleY,alpha,C,stepSize,iteratorMu,sigma))**2))
    
    derivativeCheck=(fPos-fNeg)/(2*epsilon)                       
    
    return derivativeCheck


    
        