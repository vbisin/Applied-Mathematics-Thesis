## Stochastic Gradient Descent Loop for Algorithm 

import numpy as np
from gradients import alphaGradient, WGradient
from estimateSignal import estimateSignal
#from gradientCheck import alphaGradCheck,WGradCheck
from armijo import armijoAlpha, armijoW

def multiSGDthres(x,y,alpha,W,negWTransDeriv): 
    
##Recover variable dimensions
    samples=x.shape[1]
    samplesIterator=np.arange(samples)
## Initializations 

    # Error criterion needed to exit Stochastic Gradient descent 
    threshold=.0005
    

## Graph Variables to return 
    
    #Average error over samples per epoch 
    errorEpoch=list()
    errorEpoch.append(np.mean(np.mean((x-y)**2,axis=0)))
    
    #Historical values of alpha and W
    global alphaHistory,WHistory
    alphaHistory=list()
    WHistory=list()
    alphaHistory.append(alpha)
    WHistory.append(W)
    
    # Lists containing average gradient per epoch for alpha and W
    alphaGradEpoch=list()
    WGradEpoch=list()
    
    ##Learning Rates
    global learningRatesAlpha, learningRatesW
    learningRatesAlpha=list()
    learningRatesW=list()
   
    global alphaChange, WChange
    alphaChange=alpha
    WChange=W
        
## Stochastic Gradient Descent loop, completes at least 4 epochs 
    # and exits if alpha and W grads' sum is less than the threshold
    
    while (len(errorEpoch)<3 or abs(errorEpoch[len(errorEpoch)-2]-errorEpoch[len(errorEpoch)-1])>threshold) and len(errorEpoch)<100:
 
        # Matrix to be returned after each iteration of the SGD algorithm
        # Returns the MSE error per sample, alpha gradient, and W gradient 
        returnMatrix=np.zeros((samples,3),dtype=object)
        returnMatrix=np.asarray([SGDSample(alphaChange,WChange,x[:,sample],y[:,sample],negWTransDeriv) for sample in samplesIterator])    
   
    
        alpha=alphaChange
        W=WChange
        
        # For each epoch record average error of each sample
        errorEpoch.append(np.average(np.average(returnMatrix[:,0])))
        
        # Compute average per epoch alpha and W gradients 
        alphaGradEpoch.append(np.average(np.average(returnMatrix[:,1])))
        WGradEpoch.append(np.average(np.average(returnMatrix[:,2])))
                
        print("Threshold SGD iteration: " + str(len(errorEpoch)-1))
        

    
    return (alpha, W, errorEpoch,np.array(alphaHistory),np.array(WHistory),alphaGradEpoch,WGradEpoch,learningRatesAlpha,learningRatesW)            
    
    
def SGDSample(alpha,W,sampleX,sampleY,negWTransDeriv):
    ## Initiate global variables
    global alphaChange,WChange,alphaHistory,WHistory,learningRatesAlpha,learningRatesW
    
    ##Calculate gradients for alpha and W
    alphaGrad=alphaGradient(alpha,sampleX,sampleY,W)
    WGrad=WGradient(W,sampleX,sampleY,alpha,negWTransDeriv)
    
    # Calculate optimal step sizes using the Armijo rule
    learningRateAlpha=armijoAlpha(W,sampleX,sampleY,alpha,alphaGrad)
    #learningRateW=armijoW(W,sampleX,sampleY,alpha,WGrad)
    learningRateW=.00000005   
    
    learningRatesAlpha.append(learningRateAlpha)
    learningRatesW.append(learningRateW)


    ## Update alpha and W                 
    alpha=alpha-learningRateAlpha*alphaGrad
    W=W-learningRateW*WGrad
    
    alphaChange=alpha
    WChange=W
    alphaHistory.append(alpha)
    WHistory.append(W)    
    
    # Record MSE for each sample
    errorSample=(sampleX-estimateSignal(W,sampleY,alpha))**2
    
    return (errorSample,alphaGrad,WGrad)    

