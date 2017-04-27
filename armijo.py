## Estimates the optimal step sizes for alpha and W using the Armijo rule

from estimateSignal import estimateSignal
import numpy as np

def armijoAlpha (W,sampleX,sampleY,alpha,alphaGrad):
    
    ## Initialize parameters - maximal step size is 10, iteratively decreased by 1/2
    originalStepSize=10
    beta=.5
    counter=0
    
    LHS=sum(.5*((sampleX-estimateSignal(W,sampleY,(alpha-originalStepSize*alphaGrad)))**2))
    RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha))**2))-.5*originalStepSize*((np.linalg.norm(alphaGrad))**2)
    freedom=False

    ## Followin the Armijo rule, keep decrementing the step size until the condition is reached     
    while not(LHS<=RHS):
        freedom=True
        
        counter+=1
        stepSize=originalStepSize*(beta**counter)
    
        LHS=sum(.5*((sampleX-estimateSignal(W,sampleY,(alpha-stepSize*alphaGrad)))**2))
        RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha))**2))-.5*stepSize*((np.linalg.norm(alphaGrad))**2)
            
    if freedom:
        return originalStepSize*(beta**(counter-1))
    else:
        return originalStepSize
        
def armijoW (W,sampleX,sampleY,alpha,WGrad):
    
    ## Initialize parameters - maximal step size is 10, iteratively decreased by 1/2
    originalStepSize=10
    beta=.5
    counter=0
    
    LHS=sum(.5*((sampleX-estimateSignal((W-originalStepSize*WGrad),sampleY,alpha))**2))
    RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha))**2))-.5*originalStepSize*((np.linalg.norm(WGrad))**2)
    freedom=False
   
    ## Followin the Armijo rule, keep decrementing the step size until the condition is reached     
    while not(LHS<=RHS):
        freedom=True
        counter+=1
        stepSize=originalStepSize*(beta**counter)
        
        LHS=sum(.5*(sampleX-estimateSignal((W-stepSize*WGrad),sampleY,alpha))**2)  
        RHS=sum(.5*((sampleX-estimateSignal(W,sampleY,alpha))**2))-.5*stepSize*((np.linalg.norm(WGrad))**2)
        
    if freedom:
        return originalStepSize*(beta**(counter-1))
    else:
        return originalStepSize
        