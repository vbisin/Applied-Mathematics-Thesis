
## Stochastic Gradient Descent Loop for Algorithm 


import numpy as np
from gradients import alphaGradient, wGradient
from estimateSignal import estimateSignal



def multiSGDthres(x,y,alpha,W): 
##Recover variable dimensions
    samples=x.shape[1]
    maxSamples=np.arange(samples)
## Initializations 

    #Learning Rates
    learningRateAlpha=.0001
    learningRateW=.00000001     
    

    # Function error difference between consecutive epochs (initialization is arbitrary)
    functionError=1
    
    # Divergence criterion 
    divergenceThreshold=10
    
    
    # Error criterion needed to exit Stochastic Gradient descent 
    errorThreshold=1
    

## Graph Variables to return 
    
    #Average error over samples per epoch 
    errorEpoch=list()
    errorEpoch.append(np.mean(np.mean((x-y)**2,axis=0)))
    
    #Historical values of alpha and W
    alphaHistory=list()
    WHistory=list()
    alphaHistory.append(alpha)
    WHistory.append(W)

    #Historical values of the gradients of alpha and W
    savedAlphaGrads=list()
    savedWGrads=list()
    
    #Learning Rates
    learningRates=list()
    learningRates.append(learningRateAlpha)
    learningRates.append(learningRateW)
      
        
## Stochastic Gradient Descent loop, completes at least two epochs 
    # and exits if error between epochs is less than the threshold
    
    while len(errorEpoch)<3 or functionError>errorThreshold:
        
        # Function error for each sample 
        errorSample=list()
        
    ## Iterate over samples
        for i in maxSamples:

            # Current samples
            sampleX=x[:,i]
            sampleY=y[:,i]
            
            
        ##Calculate gradients for alpha and W
            alphaGrad=alphaGradient(sampleX,sampleY,alpha,W)
            WGrad=wGradient(sampleX,sampleY,alpha,W)
          
            #Record current alpha and W gradients
            savedAlphaGrads.append(alphaGrad)
            savedWGrads.append(WGrad)
            

        ## Update alpha and W                 
            alpha=alpha-learningRateAlpha*alphaGrad
            W=W-learningRateW*WGrad
            
            # Calculate current objective function estimate 
            estimate=estimateSignal(W,sampleY,alpha)
            
            # Record MSE for each sample
            errorSample.append(np.mean((sampleX-(estimate))**2))
            
            # Record current alpha and W values
            alphaHistory.append(alpha)
            WHistory.append(W)
            
   
        # For each epoch record average error of each sample
        errorEpoch.append(np.average(errorSample))
        
        # Update function error between consecutive epochs 
        functionError=abs(errorEpoch[len(errorEpoch)-1]-errorEpoch[len(errorEpoch)-2])
        
        print("Threshold SGD " + str(len(errorEpoch)-1))
        
        
   ## Divergence criterion     
       # If algorithm has completed more than 2 epochs and is greater than the divergence 
       # threshold then exit 
        if len(errorEpoch)>2 and functionError>divergenceThreshold:
            print("diverged")
            break
                
        
        
    return (alpha, W, errorEpoch,errorSample,alphaHistory,WHistory,learningRates,savedAlphaGrads,savedWGrads)            
            
                        
                        
 

                        
 
