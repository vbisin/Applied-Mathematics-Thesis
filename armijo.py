'''This script describes two function armijoAlpha and armijoW that estimate the optimal step sizes 
for alpha and W for one signal using the Armijo rule
(see https://en.wikipedia.org/wiki/Backtracking_line_search), respectively'''

from estimateSignal import estimateSignal
import numpy as np

def armijoAlpha (W,sampleX,sampleY,alpha,alphaGrad,C,stepSize,iteratorMu,sigma):
    
    ## Initialize parameters - maximal step size is originalLearningRate,
    ##iteratively multiplied (decreased) by beta<1
    originalLearningRate=1
    beta=.1
    counter=-1
    # Initialize so we enter the while loop
    LHS=1
    RHS=0
    
    
    ## Following the Armijo rule, keep decrementing the learning rate until the condition is reached     
    while not(LHS<=RHS):
        
        # iteratively decrease the learning rate by beta 
        counter+=1
        learningRate=originalLearningRate*(beta**counter)
        
        ## These are the LHS and RHS of the Armijo condition
        LHS=sum(.5*((sampleX-estimateSignal(W,sampleY,(alpha-learningRate*alphaGrad),C,stepSize,iteratorMu,sigma))**2))
        RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha,C,stepSize,iteratorMu,sigma))**2))-.3*learningRate*((np.linalg.norm(alphaGrad))**2)
        

    return learningRate

    
def armijoW (W,sampleX,sampleY,alpha,WGrad,C,stepSize,iteratorMu,sigma):
    
    ## Initialize parameters - maximal step size is originalLearningRate,
    ##iteratively (decreased) by beta<1
    originalLearningRate=1
    beta=.1    
    counter=-1
    # Initialize so we enter the while loop
    LHS=1
    RHS=0
    
    ## Following the Armijo rule, keep decrementing the learning rate until the condition is reached     
    while not(LHS<=RHS):
        
        # iteratively decrease the learning rate by beta 
        counter+=1
        learningRate=originalLearningRate*(beta**counter)
        
        ## These are the LHS and RHS of the Armijo condition
        LHS=sum(.5*((sampleX-estimateSignal((W-learningRate*WGrad),sampleY,alpha,C,stepSize,iteratorMu,sigma))**2))  
        RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha,C,stepSize,iteratorMu,sigma))**2))-.3*learningRate*((np.linalg.norm(WGrad))**2)
        
       
    return learningRate



