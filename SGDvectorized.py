## Stochastic Gradient Descent Loop for Algorithm 

import numpy as np
from gradients import alphaGradient, WGradient
from estimateSignal import estimateSignal
from gradientCheck import alphaGradCheck,WGradCheck


def multiSGDthres(x,y,alpha,W): 
    
##Recover variable dimensions
    samples=x.shape[1]
    samplesIterator=np.arange(samples)
## Initializations 

    #Learning Rates
    learningRateAlpha=.000005
    learningRateW=.000005     
    
    # Divergence criterion 
    divergenceThreshold=10
    
    
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
    
    #Learning Rates
    learningRates=list()
    learningRates.append(learningRateAlpha)
    learningRates.append(learningRateW)
      
    global alphaChange, WChange
    alphaChange=alpha
    WChange=W
        
## Stochastic Gradient Descent loop, completes at least two epochs 
    # and exits if alpha and W grads' sum is less than the threshold
    
    while (len(errorEpoch)<3 or (abs(alphaGradEpoch[len(alphaGradEpoch)-1])+abs(WGradEpoch[len(WGradEpoch)-1]))>threshold) and len(errorEpoch)<30:
 
        # Matrix to be returned after each iteration of the SGD algorithm
        # Returns the MSE error per sample, alpha gradient, and W gradient 
        returnMatrix=np.zeros((samples,3),dtype=object)
        returnMatrix=np.asarray([samplesSGDLoop(alphaChange,WChange,x[:,sample],y[:,sample],learningRateAlpha,learningRateW) for sample in samplesIterator])    
   
    
        alpha=alphaChange
        W=WChange
        # For each epoch record average error of each sample
        errorEpoch.append(np.average(np.average(returnMatrix[:,0])))
        
        # Compute average per epoch alpha and W gradients 
        alphaGradEpoch.append(np.average(np.average(returnMatrix[:,1])))
        WGradEpoch.append(np.average(np.average(returnMatrix[:,2])))
        
        # Update function error between consecutive epochs 
        functionError=errorEpoch[len(errorEpoch)-1]-errorEpoch[len(errorEpoch)-2]
        
        print("Threshold SGD " + str(len(errorEpoch)-1))
        
        
   ## Divergence criterion     
       # If algorithm has completed more than 2 epochs and is greater than the divergence 
       # threshold then exit 
        if len(errorEpoch)>3 and functionError>divergenceThreshold:
            print("diverged")
            break
                
        
        
    return (alpha, W, errorEpoch,np.array(alphaHistory),np.array(WHistory),learningRates,alphaGradEpoch,WGradEpoch)            
    
    
def samplesSGDLoop(alpha,W,sampleX,sampleY,learningRateAlpha,learningRateW):
         

    ##Calculate gradients for alpha and W
    alphaGrad=alphaGradient(sampleX,sampleY,alpha,W)
    WGrad=WGradient(sampleX,sampleY,alpha,W)
    
    #Check alpha and W gradients using def. of derivative 
    
    #alphaGradCheck(sampleX,sampleY,alpha,W)
    #checker=WGradCheck(sampleX,sampleY,alpha,W)
    #print("The difference in W grad's summed up is: "+str(sum(sum(checker))))
   
    ## Update alpha and W                 
    alpha=alpha-learningRateAlpha*alphaGrad
    W=W-learningRateW*WGrad
    
    global alphaChange,WChange,alphaHistory,WHistory
    alphaChange=alpha
    WChange=W
    alphaHistory.append(alpha)
    WHistory.append(W)    
    # Record MSE for each sample
    errorSample=(sampleX-estimateSignal(W,sampleY,alpha))**2
    
    return (errorSample,alphaGrad,WGrad)
    
    
     

                        
 
