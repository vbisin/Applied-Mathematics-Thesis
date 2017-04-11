## Checks accuracy of gradients for both alpha and W

import numpy as np
from estimateSignal import estimateSignal
from gradients import alphaGradient, WGradient
import copy
#import random

def alphaGradCheck(sampleX,sampleY,alpha,W):
    
    alphaDim=len(alpha)
    alphaIterator=np.arange(alphaDim)
    # choose vector entry
    alphaSum=0
    for i in alphaIterator:
        
        alphaGrad=alphaGradient(sampleX,sampleY,alpha,W)[i]
        
        
        epsilon=.0001
        alphaEpsPos=copy.deepcopy(alpha)
        alphaEpsPos[i]=alphaEpsPos[i]+epsilon
        alphaEpsNeg=copy.deepcopy(alpha)         
        alphaEpsNeg[i]=alphaEpsNeg[i]-epsilon
        
        fPos=sum(.5*((sampleX-estimateSignal(W,sampleY,alphaEpsPos))**2))
        fNeg=sum(.5*((sampleX-estimateSignal(W,sampleY,alphaEpsNeg))**2))
        
        derivativeCheck=(fPos-fNeg)/(2*epsilon)          
        bools=True
        
           
        if abs(alphaGrad-derivativeCheck)>.01:
            print("Alpha grad's "+str(i+1)+"th entry is incorrect. The difference between the two is: " + str(abs(alphaGrad-derivativeCheck)))
            bools=False
        
        alphaSum=abs(alphaGrad)+alphaSum
            
    if bools==True:
            print("Alpha gradient is correct (and sum over alphas is: " + str(alphaSum)+")")
                   
             
def WGradCheck(sampleX,sampleY,alpha,W):
        
    WIterator=np.arange(len(sampleX))
    checkableW=np.zeros((len(sampleX),len(sampleX)))
# choose matrix entry
    checkableW=np.asarray([np.asarray([speedWGradCheck(i,j,sampleX,sampleY,alpha,W) for j in WIterator]).astype(np.float) for i in WIterator])
    gradDifference=checkableW-WGradient(sampleX,sampleY,alpha,W)
    
    if (abs(gradDifference)<.01).all():
        print("W Gradient is correct")
        return gradDifference
    else:    
        return gradDifference      

 
## Sub function to speed up WGradCheck
def speedWGradCheck(i,j,sampleX,sampleY,alpha,W):    
    
    epsilon=.0001
    WEpsPos=copy.deepcopy(W)
    WEpsPos[i][j]=WEpsPos[i][j]+epsilon
    WEpsNeg=copy.deepcopy(W)
    WEpsNeg[i][j]=WEpsPos[i][j]-epsilon
    
    
    
    fPos=sum(.5*((sampleX-estimateSignal(WEpsPos,sampleY,alpha))**2))
    fNeg=sum(.5*((sampleX-estimateSignal(WEpsNeg,sampleY,alpha))**2))
    
    derivativeCheck=(fPos-fNeg)/(2*epsilon)                       
    
    return derivativeCheck
        
        
    
       
       



   
         