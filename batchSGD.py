
'''Mini-Batch Stochastic Gradient Descent Loop for Algorithm '''

import numpy as np
from gradients import alphaGradient, WGradient
from estimateSignal import estimateSignal
from armijo import armijoAlpha, armijoW

def batchSGD(x,y,alpha,W,stepSize,iteratorMu,WRandomInitialization,WSemiWellInitialized,\
             WWellInitialized,alphaRandomInitialization,alphaOnesIntialized,alphaWellInitialized,armijo,\
             sigma,C,momentumConstant): 
  
    print("x) running miniBatch SGD")
    
    ## Parameters to keep track of changes in alpha and W
    global alphaHistory,WHistory, alphaChange, WChange
    alphaHistory=list()
    WHistory=list()
    alphaEpoch=list()
    WEpoch=list() 
    alphaChange=alpha
    WChange=W
    
    ## Weight update (for momentum term)
    global weightUpdateAlpha, weightUpdateW
    weightUpdateAlpha=list()
    weightUpdateW=list()
    weightUpdateAlpha.append(0)
    weightUpdateW.append(0)
    
    
    ## Determine whether to minimize alpha, W, or both. If either W or alpha are well-initialized, 
    ## we don't optimize those values
    if (WRandomInitialization==True or WSemiWellInitialized==True) and (alphaRandomInitialization==True or alphaOnesIntialized==True):
        minimizeBoth=True
        minimizeW=False
        minimizeAlpha=False
        print('xi) Minimizing alpha, W')
    
        #Initialize alpha and W historical values         
        alphaEpoch.append(alpha)
        WEpoch.append(W)

    elif (WRandomInitialization==True or WSemiWellInitialized==True): 
        minimizeBoth=False
        minimizeW=True
        minimizeAlpha=False  
        print('xi) Minimizing W')
    
        # Initialize W historical values    
        WEpoch.append(W)    
        
    elif (alphaRandomInitialization==True or alphaOnesIntialized==True):
        minimizeBoth=False
        minimizeW=False
        minimizeAlpha=True
        print('xi) Minimizing alpha')
    
        #initialize alpha historical values    
        alphaEpoch.append(alpha)
    else:
        print(" \n \n Pirla, alpha and W are already well initialized, you are not \
        minimizing over anything \n \n ")

    
    if armijo:
        print("xii) Step size optimizing using Armijo")
    else:
        print("xii) Step size constant")
    
        

## SGD algorithm initialization
 
    # Error criterion needed to exit stochastic gradient descent loop
    threshold=.0005

    # Maximum number of iterations for stochastic gradient descent loop
    maxIterations=500
    
    ##Recover variable dimensions
    numExamples=x.shape[1]
    iteratorSamples=np.arange(numExamples)



    ## Initialize mini-batch size
    batchSize=1
    numBatches=numExamples/batchSize
    batchesIterator=np.linspace(batchSize,numExamples,numBatches,dtype=int) 
    print("xiii) Batch Size is " +str(batchSize))
    
    
    #If the algorithm begins to oscillate we increase the batch size, here we define the other 
    # possible batch sizes
    batchSizes=list()
    possibleBatchSizes=np.array([1,50,500,1000,5000,10000])
    batchCounter=1
    batchSizeGapCounter=0
    
    # Find possible batch sizes by determining if they are divisors of the total number of samples
    for i in possibleBatchSizes:
        if numExamples%i==0:
            batchSizes.append(i)
    
        
    

    # Initialize sum of squared residuals (SSR) average error over samples per epoch 
    SSREpochAvg=list()
    SSROriginalParameters=np.asarray([.5*(sum((x[:,sample]-estimateSignal(W,y[:,sample],alpha,C,stepSize,iteratorMu,sigma))**2)) for sample in iteratorSamples])
    SSROriginalParametersAverage=np.mean(SSROriginalParameters)
    SSREpochAvg.append(SSROriginalParametersAverage)

    print("\n \n Average MSE with original noise is: "+str(.5*np.mean(np.sum((x-y)**2,axis=0))))
    print("Average MSE with original parameters is: "+str(SSROriginalParametersAverage))


    # Lists containing average gradient per epoch for alpha and W
    alphaGradEpoch=list()
    WGradEpoch=list()
    
    
    #Average Learning Rates per Epoch
    learningRateAlphaEpoch=list()
    learningRateWEpoch=list()
   
   

    
## Stochastic Gradient Descent loop


    # SGD stops if average SSR is less than the threshold or the number of iterations is over the max
    while len(SSREpochAvg)<3 or (abs(SSREpochAvg[len(SSREpochAvg)-3]-SSREpochAvg[len(SSREpochAvg)-1])>threshold and (abs(SSREpochAvg[len(SSREpochAvg)-3]-SSREpochAvg[len(SSREpochAvg)-2])>threshold) and len(SSREpochAvg)-1<maxIterations):
    
        
        
        ## Determine if SGD algorithm is oscillating, if it is, increase the mini-batch size    
        
        # Between any batch increase have at least 10 iterations 
        if batchSizeGapCounter>10:
            # From SSR average per epoch, determines if algorithm is oscillating 
            if len(SSREpochAvg)>20 and SSREpochAvg[len(SSREpochAvg)-1]>SSREpochAvg[len(SSREpochAvg)-8] and SSREpochAvg[len(SSREpochAvg)-2]>SSREpochAvg[len(SSREpochAvg)-6]and batchCounter<(len(batchSizes)):
                # determine the new batch size 
                batchSize=possibleBatchSizes[batchCounter]
                numBatches=numExamples/batchSize
                batchesIterator=np.linspace(batchSize,numExamples,numBatches,dtype=int) 
                print("Because of oscillation, batch size has increased from "+str(possibleBatchSizes[batchCounter-1])+" to "+ str(possibleBatchSizes[batchCounter]))
                batchCounter+=1
                batchSizeGapCounter=0
        else:
            batchSizeGapCounter+=1
            
        # Matrix to be returned after each iteration of the SGD algorithm
        # Returns the SSR error per sample and variable gradient(s) (alpha,W, or both)
        returnMatrix=np.zeros((numBatches,2),dtype=object)
        returnMatrix=np.asarray([batchSGDSample(x[:,(batchSample-batchSize):batchSample],y[:,(batchSample-batchSize):batchSample],stepSize,\
                                iteratorMu,minimizeBoth,minimizeAlpha,minimizeW,armijo,sigma,C,momentumConstant) for batchSample in batchesIterator])    
        
        
        # Compute sample average per epoch alpha/W gradients & learning rates  
        if minimizeAlpha:
            alphaGradEpoch.append(np.linalg.norm(np.average(returnMatrix[:,0])))
            learningRateAlphaEpoch.append(np.mean(returnMatrix[:,1]))
           
            alphaEpoch.append(np.mean(np.asarray(alphaHistory),axis=0))
            alphaHistory=list()
            
           
        elif minimizeW:
            WGradEpoch.append(np.linalg.norm(np.average(returnMatrix[:,0])))
            learningRateWEpoch.append(np.mean(returnMatrix[:,1]))
            
            WEpoch.append(np.mean(np.asarray(WHistory),axis=0))
            WHistory=list()
          
            
        elif minimizeBoth: 
            alphaGradEpoch.append(np.linalg.norm(np.average(returnMatrix[:,0])))
            WGradEpoch.append(np.linalg.norm(np.average(returnMatrix[:,1])))
            learningRateAlphaEpoch.append(np.mean(returnMatrix[:,2]))
            learningRateWEpoch.append(np.mean(returnMatrix[:,3]))
            
            alphaEpoch.append(np.mean(np.asarray(alphaHistory),axis=0))
            alphaHistory=list()
            WEpoch.append(np.mean(np.asarray(WHistory),axis=0))
            WHistory=list()
           
         # For each epoch record average SSR error of each sample
        errorSampleEpoch=np.asarray([.5*(sum((x[:,sample]-estimateSignal(WEpoch[len(WEpoch)-1],y[:,sample],alphaEpoch[len(alphaEpoch)-1],C,stepSize,iteratorMu,sigma))**2)) for sample in iteratorSamples])
        SSREpochAvg.append(np.average(errorSampleEpoch))
        
          
        
            ## Print how the algorithm is doing 
        if SSREpochAvg[len(SSREpochAvg)-1] > SSREpochAvg[len(SSREpochAvg)-2]:      
            print("Threshold SGD epoch iteration: " + str(len(SSREpochAvg)-1) + ", with epoch average MSE "+str(SSREpochAvg[len(SSREpochAvg)-1])+" (INCREASE)")

        else:
            print("Threshold SGD epoch iteration: " + str(len(SSREpochAvg)-1) + ", with epoch average MSE "+str(SSREpochAvg[len(SSREpochAvg)-1])+" (DECREASE)")

        
        
    return (alphaChange, WChange, SSREpochAvg,alphaEpoch,WEpoch, alphaGradEpoch,WGradEpoch,learningRateAlphaEpoch,learningRateWEpoch)            
    
    
def batchSGDSample(batchX,batchY,stepSize,iteratorMu,minimizeBoth,minimizeAlpha,minimizeW,armijo,sigma,C,momentumConstant):
    ## Initiate global variables
    global alphaChange,WChange,alphaHistory,WHistory,learningRatesAlpha,learningRatesW,weightUpdateAlpha, weightUpdateW
      
    
    
    batchSize=batchX.shape[1]
    batchIterator=np.arange(batchSize)
    
    ## Minimizing over alpha 
    if minimizeAlpha:

        # Calculate derivative
        alphaGrads=np.transpose(np.asarray([alphaGradient(alphaChange,batchX[:,sample],batchY[:,sample],WChange,C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
        alphaGrad=np.mean(alphaGrads,axis=1)
        
        
        # Learning Rates
        if armijo:
            learningRateAlphas=(np.asarray([armijoAlpha(WChange,batchX[:,sample],batchY[:,sample],alphaChange,alphaGrads[:,sample],C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
            learningRateAlpha=np.mean(learningRateAlphas)
            
        else:
            learningRateAlpha=.0005 
        
        # Update alpha 
        pastWeightAlpha=weightUpdateAlpha.pop()
        weightUpdateAlpha.append(alphaChange-learningRateAlpha*alphaGrad)
        
        alphaChange=alphaChange-learningRateAlpha*alphaGrad-momentumConstant*pastWeightAlpha
        
        # Update historical values
        alphaHistory.append(alphaChange) 
        

        return (alphaGrad,learningRateAlpha)
    
    
    
    ## Minimizing over W and sigma
    elif minimizeW:
        
         # Calculate derivative
         WGrads=np.transpose(np.array([WGradient(WChange,batchX[:,sample],batchY[:,sample],alphaChange,C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
         WGrad=np.mean(WGrads,axis=1)
        
         # Learning Rates
         if armijo: 
             learningRateWs=(np.array([armijoW(WChange,batchX[:,sample],batchY[:,sample],alphaChange,WGrads[:,sample],C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
             learningRateW=np.mean(learningRateWs)
             
         else:
             learningRateW=.000000000000000005

       
        # Update W and sigma
         pastWeightW=weightUpdateW.pop()
         weightUpdateW.append(WChange-learningRateW*WGrad)
      
         
         WChange=WChange-learningRateW*WGrad-momentumConstant*pastWeightW
         
         # Update historical values
         WHistory.append(WChange)

         return  (WGrad,learningRateW)



    ## If minimizing over  alpha, W, and sigma
    elif minimizeBoth:
        
        # Calculate derivatives
        
        #alpha
        alphaGrads=np.transpose(np.asarray([alphaGradient(alphaChange,batchX[:,sample],batchY[:,sample],WChange,C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
        alphaGrad=np.mean(alphaGrads,axis=1)       
        
        
        #W
        WGrads=np.transpose(np.array([WGradient(WChange,batchX[:,sample],batchY[:,sample],alphaChange,C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
        WGrad=np.mean(WGrads,axis=1)
        
   
         
        # Learning Rates
        if armijo:

            # learning rate alpha
            learningRateAlphas=(np.asarray([0 if sum(alphaGrads[:,sample])==0 else armijoAlpha(WChange,batchX[:,sample],batchY[:,sample],alphaChange,alphaGrads[:,sample],C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
            learningRateAlpha=np.mean(learningRateAlphas)
            
            #learning rate W
            learningRateWs=(np.array([0 if sum(WGrads[:,sample])==0 else armijoW(WChange,batchX[:,sample],batchY[:,sample],alphaChange,WGrads[:,sample],C,stepSize,iteratorMu,sigma) for sample in batchIterator]))
            # average learning rate taking away the zero values
            learningRateW=learningRateWs.sum()/(learningRateWs!=0.).sum()
            if np.linalg.norm(WGrad)>20000:
                learningRateW=learningRateW/10.

            
        else:
            learningRateAlpha=.0005   
            learningRateW=.00000000005 

        
        #Update Alpha and W
        pastWeightAlpha=weightUpdateAlpha.pop()
        pastWeightW=weightUpdateW.pop()


        weightUpdateAlpha.append(alphaChange-learningRateAlpha*alphaGrad)
        weightUpdateW.append(WChange-learningRateW*WGrad)
        
        alphaChange=alphaChange-learningRateAlpha*alphaGrad-momentumConstant*pastWeightAlpha
        WChange=WChange-learningRateW*WGrad-momentumConstant*pastWeightW
         

        # Update historical values 
        alphaHistory.append(alphaChange)  
        WHistory.append(WChange)   

     
    
        return (alphaGrad,WGrad,learningRateAlpha,learningRateW)    
    