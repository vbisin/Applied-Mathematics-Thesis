''' Driver of algorithm based on the Vittorio Bisin's Master's thesis: 
    "A Study in Total Variation Denoising and its Validity in a Recently Proposed Denoising Algorithm." 
    
    The algorithm denoises randomly generated piecewise constant functions by learning 
    an influence function and a linear filter.'''

import numpy as np
from createStep import createSteps
import matplotlib.pyplot as plt
from rbf import rbfF
import os
from estimateSignal import estimateSignal
import random
from rbfCenters import GRBFCenters
import copy
from batchSGD import batchSGD

## Define starting parameters for algorithm 

# Running algorithm on HPC
HPC=False


# How to initialize W
WRandomInitialization=False
WSemiWellInitialized=True
WWellInitialized=False


# How to initialize alpha 
alphaRandomInitialization=False
alphaOnesIntialized=True
alphaWellInitialized=False


# Initialize sigma (standard deviation of Gaussian RBF functions)
sigma=1


#Initialize whether to optimize using Armijo Backtracking or use a constant step size
armijo=True

#Momentum Constant
momentumConstant=0

#Minimum original jump size
minJumpSize=25

#Number of randomly generated signals to train over
numberExamples=5000

#Length of each signal
signalLength=30

# Max value in range of each signal
maxValue=60

# Initialize distance between each center in the RBF function
stepSize=.5

#Standard deviation of added noise 
noiseStdDev=5

# print important properties of algorithm iteration
print ("This algorithm iteration has the following properties: \n i) number of examples is: "+str(numberExamples)+ \
      "\n ii) signal length is: "+str(signalLength)+ "\n iii) std. deviation of noise is: " +str(noiseStdDev))


#############################################

#Retrive randomly generated piecewise constant signals (x is original, y is noisy)
(x,y,numberJumpsTrain)=createSteps(numberExamples,signalLength,maxValue,noiseStdDev,minJumpSize)

## Initialize W

# Random initialization 
if WRandomInitialization:
    print("iv) W randomly initialized in [-5,5]") 
    W=np.asarray([random.uniform(-5, 5) for j in range(2)]).astype(np.float) 
    WInitial="WRandom"

# W close to optimally initialized
elif WSemiWellInitialized:
    print("iv) W nearly optimally initialized") 
    W=[1.5,-1.5]
    WInitial="WSemiOpt"
                
# W optimally initialized to finite difference operator    
elif WWellInitialized:
    W=np.zeros((signalLength,signalLength))
    print("iv) W optimally initialized") 
    W=[1.0,-1.0]
    WInitial="WOpt"



# Retrieve RBF centers/step sizes and iterator array 

iteratorSamples=np.arange(numberExamples)
noisyDerivs=np.transpose(np.asarray([np.convolve(W,y[:,sample],'same') for sample in iteratorSamples]))
alphaDim,iteratorMu=GRBFCenters(noisyDerivs,stepSize)


print('v) sigma value is: '+str(sigma))
print ("vi) the range of finite differences is ["+str(int(np.min(noisyDerivs)))+","+str(int(np.max(noisyDerivs)))+"]")
print("vii) alpha dimension is: " +(str(alphaDim)+" (step size is "+str(stepSize)+")"))
   

## Alpha Intialization
    
# alpha randomly initialized
if alphaRandomInitialization:
    minA=0
    maxA=10
    print('viii) alphas randomly initialized in ['+str(minA)+','+str(maxA)+']')         
    alpha=np.asarray([random.uniform(minA, maxA) for j in range(alphaDim)]).astype(np.float)
    alphaInitial="alphaRand"       


#alpha constantly intialized at ones
elif alphaOnesIntialized:
    print("viii) alphas initialized to ones")
    alpha=np.ones(alphaDim)
    alphaInitial="alphaZero"       


# attempted optimized alpha
elif alphaWellInitialized: 
     print("viii) alphas optimally initialized") 
     alpha=copy.deepcopy(iteratorMu)
     counter=0
     for entry in alpha:
        if abs(entry)<5:
            alpha[counter]=alpha[counter]*0
        elif (abs(entry)<10):
            alpha[counter]=alpha[counter]*0
        elif (abs(entry)<20):
             alpha[counter]=alpha[counter]*.75
        else:
            alpha[counter]=alpha[counter]*.8
        
        counter+=1
     alphaInitial="alphaOpt"       

  
#Tag describing type of experiment undergone, for graph names
experimentTag=str(alphaInitial)+str(WInitial)+str(int(sigma))+str(int(numberExamples))+str(int(stepSize))


# Calculate cumulative matrix
C=np.zeros((signalLength,signalLength))
for row in range(signalLength):
    for column in range(row+1):
        C[row,column]=1
         
         
#######################################################

         

## Run mini-batch Stochastic Gradient Descent Algorithm 
(alpha, W, SSREpochAvg,alphaEpoch,WEpoch, alphaGradEpoch,WGradEpoch,learningRateAlphaEpoch,learningRateWEpoch)=batchSGD(x,y,alpha,W,\
stepSize,iteratorMu,WRandomInitialization,WSemiWellInitialized,WWellInitialized,alphaRandomInitialization,alphaOnesIntialized,alphaWellInitialized,\
armijo,sigma,C,momentumConstant)




###########################################################

## Train Set Evaluation 

# Estimate  signals for all samples
predictedSignals=np.transpose(np.asarray([estimateSignal(W,y[:,sample],alpha,C,stepSize,iteratorMu,sigma) for sample in iteratorSamples]).astype(np.float))


## Print statement if the algorithm denoised the signal (and calculate percent denoised)
if sum(sum(np.abs(x-y)))>sum(sum(np.abs(x-predictedSignals))):
    print ("Denoising did occur")
    print("Percent denoised: "+str(100*(1-(sum(sum(np.abs(x-predictedSignals)))/sum(sum(np.abs(x-y)))))))
    print("original SAV error: " +str(sum(sum(np.abs(x-y)))))
    print("current SAV error: " +str(sum(sum(np.abs(x-predictedSignals)))))
    
else: 
    print("denoising did not occur")
    print("original SAV error: " +str(sum(sum(np.abs(x-y)))))
    print("current SAV error: " +str(sum(sum(np.abs(x-predictedSignals)))))


print('Sum of Squared Residuals on train set was'+str(SSREpochAvg[len(SSREpochAvg)-1]))




#########################################################

##Test Set Evaluation

# Create test set (half size as training set)
sizeTestSet=numberExamples/2
iteratorTestSamples=np.arange(sizeTestSet)
(xTest,yTest,numberJumpsTest)=createSteps(sizeTestSet,signalLength,maxValue,noiseStdDev,minJumpSize)


# Calculate predicted values
predictedTestSignals=np.transpose(np.asarray([estimateSignal(W,yTest[:,sample],alpha,C,stepSize,iteratorMu,sigma) for sample in iteratorTestSamples]).astype(np.float))


## Print statement if the algorithm denoised the signal (and calculate percent denoised)
if sum(sum(np.abs(xTest-yTest)))>sum(sum(np.abs(xTest-predictedTestSignals))):
    print ("Test Denoising did occur")
    print("Test Percent denoised: "+str(100*(1-(sum(sum(np.abs(xTest-predictedTestSignals)))/sum(sum(np.abs(xTest-yTest)))))))
    print("Test original error:" +str(sum(sum(np.abs(xTest-yTest)))))
    print("Test current error:" +str(sum(sum(np.abs(xTest-predictedTestSignals)))))
    
else: 
    print("Test denoising did not occur")
    print("Test original error:" +str(sum(sum(np.abs(xTest-yTest)))))
    print("Test current error:" +str(sum(sum(np.abs(xTest-predictedTestSignals)))))


# Calculate SSR for test set
epochsArray=np.arange(len(SSREpochAvg))
if alphaEpoch and WEpoch:
    SSRsquaredErrTest=np.asarray([np.asarray([.5*(sum((xTest[:,sample]-estimateSignal(WEpoch[epoch],yTest[:,sample],alphaEpoch[epoch],C,stepSize,iteratorMu,sigma))**2)) for sample in iteratorTestSamples]) for epoch in epochsArray])
elif alphaEpoch:
    SSRsquaredErrTest=np.asarray([np.asarray([.5*(sum((xTest[:,sample]-estimateSignal(W,yTest[:,sample],alphaEpoch[epoch],C,stepSize,iteratorMu,sigma))**2)) for sample in iteratorTestSamples]) for epoch in epochsArray])
elif WEpoch:
    SSRsquaredErrTest=np.asarray([np.asarray([.5*(sum((xTest[:,sample]-estimateSignal(WEpoch[epoch],yTest[:,sample],alpha,C,stepSize,iteratorMu,sigma))**2)) for sample in iteratorTestSamples]) for epoch in epochsArray])

# mean SSR per sample on test set 
SSRTest=np.mean(SSRsquaredErrTest,axis=1)

print("Sum of Squared Residuals on test set is: " + str(SSRTest[len(SSRTest)-1]))


#######################################################  

## Graphs



#Directory when running code on HPC
if HPC:
    HPCDirectory="/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/kernelHeat"
    plt.switch_backend('agg')
    
    
# Graph Parameters and arrays 
iteratorAlpha=np.arange(alphaDim)
iteratorN=np.arange(signalLength)


#Best and worst train signals to graph 
exemplarySignal=0
bestNoiseDiff=100000
worstSignal=0
worstNoiseDiff=0

for signal in iteratorSamples:
    noiseDiff=sum(abs(x[:,signal]-predictedSignals[:,signal]))
    if bestNoiseDiff>noiseDiff:
        exemplarySignal=signal
        bestNoiseDiff=noiseDiff
    elif worstNoiseDiff<noiseDiff:
        worstSignal=signal
        worstNoiseDiff=signal
        
#Best and worst test signals to graph        
exemplaryTestSignal=0
bestNoiseDiff=100000
worstTestSignal=0
worstNoiseDiff=0

for signal in iteratorTestSamples:
    noiseDiff=sum(abs(xTest[:,signal]-predictedTestSignals[:,signal]))
    if bestNoiseDiff>noiseDiff:
        exemplaryTestSignal=signal
        bestNoiseDiff=noiseDiff
    elif bestNoiseDiff<noiseDiff:
        worstTestSignal=signal
        worstNoiseDiff=signal




# Plot Gaussian RBF functions for exemplary train signal    
plt.figure(1)
gaussMatrix=np.asarray([rbfF(W,y[:,exemplarySignal],alphaDim,stepSize,iteratorMu,sigma)[j,:] for j in iteratorN])
for i in range(len(gaussMatrix)):
    plt.plot(iteratorMu[110:240],gaussMatrix[i,110:240])  
plt.title("Zoomed in Gaussian Basis Functions for Signal " +str(exemplarySignal+1))
plt.xlabel('Range of Signal')
plt.ylabel('Value')   
plt.savefig('basisFunctionsZoom'+experimentTag)
plt.show     



# Final estimate of alphas
plt.figure(2)
plt.plot(iteratorMu,alpha,'b',label='Alphas')
plt.title("RBF Parameter Value: Alpha")
plt.xlabel('Gaussian RBF Center')
plt.ylabel('Alpha Value')
if HPC:
  plt.savefig(HPCDirectory+'alphas'+experimentTag)  
else:  
    plt.savefig('alphas'+experimentTag)
    plt.show





# Bar plot of learned kernel (W)
if WGradEpoch:
    fig,ax=plt.subplots()
    index=np.array([1,2])
    bar_width = 0.35
    opacity = 0.8
    wlearned=plt.bar(index,W[::-1], bar_width,label='W Kernel')
    wtheoretical=plt.bar(index+bar_width,[-1,1],bar_width,label='Finite Difference Operator')
    #plt.subplots([1,2],W,label='W kernel')
    #plt.subplots([1,2],[1,-1],label='Finite Difference Operator')
    plt.xticks([1,2])
    plt.xlabel('Entry')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
    plt.title("Learned Kernel")
    if HPC:
      plt.savefig(HPCDirectory+'kernel'+experimentTag)  
    else:  
        plt.savefig('kernel'+experimentTag)
        plt.show()




# Best original, noisy, and predicted tain signals 
plt.figure(3)
plt.title("Original, Noisy, and Predicted Values for Signal: " +str(exemplarySignal+1) +" (of "+str(numberExamples)+")")
plt.plot(iteratorN+1,x[:,exemplarySignal],'b',label='Original Signal')
plt.plot(iteratorN+1,y[:,exemplarySignal],'r',label='Noisy Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN+1,predictedSignals[:,exemplarySignal],'k',label='Predicted Signal')
plt.legend(bbox_to_anchor=(.52, .34), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'bestTrainEstimates'+experimentTag)  
else:  
    plt.savefig('bestTrainEstimates'+experimentTag)
    plt.show




# WORST Original, noisy, and predicted train signals 
plt.figure(4)
plt.title("Original, Noisy, and Predicted Values for Signal: " +str(worstSignal+1) +" (of "+str(numberExamples)+")")
plt.plot(iteratorN+1,x[:,worstSignal],'b',label='Original Signal')
plt.plot(iteratorN+1,y[:,worstSignal],'r',label='Noisy Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN+1,predictedSignals[:,worstSignal],'k',label='Predicted Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'worstTrainEstimates'+experimentTag)  
else:  
    plt.savefig('worstTrainEstimates'+experimentTag)
    plt.show





# Best Original, noisy, and predicted test signals 
plt.figure(5)
plt.title("Original, Noisy, and Predicted Values for Test Signal: " +str(exemplaryTestSignal+1)+" (of "+str(sizeTestSet)+")")
plt.plot(iteratorN+1,xTest[:,exemplaryTestSignal],'b',label='Original Test Signal')
plt.plot(iteratorN+1,yTest[:,exemplaryTestSignal],'r',label='Noisy Test Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN+1,predictedTestSignals[:,exemplarySignal],'k',label='Predicted Test Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'bestTestEstestimates'+experimentTag)  
else:  
    plt.savefig('bestTestEstestimates'+experimentTag)
    plt.show




# Worst Original, noisy, and predicted test signals 
plt.figure(6)
plt.title("Original, Noisy, and Predicted Values for Test Signal: " +str(worstTestSignal+1)+" (of "+str(sizeTestSet)+")")
plt.plot(iteratorN+1,xTest[:,worstTestSignal],'b',label='Original Test Signal')
plt.plot(iteratorN+1,yTest[:,worstTestSignal],'r',label='Noisy Test Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN+1,predictedTestSignals[:,worstTestSignal],'k',label='Predicted Test Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'WorstTestEstestimates'+experimentTag)  
else:  
    plt.savefig('WorstTestEstestimates'+experimentTag)
    plt.show



if alphaGradEpoch:
    # Average Alpha gradient per epoch on logarithmic scale
    plt.figure(7)
    plt.title("Logarithm of the Norm of the Average Alpha Gradient per Epoch")
    plt.plot(np.log(alphaGradEpoch),'b')
    plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
    plt.xlabel('Epoch')
    plt.ylabel('Natural Logarithm Value')
    if HPC:
        plt.savefig(HPCDirectory+'AlphaGrads'+experimentTag)  
    else:  
        plt.savefig('AlphaGrads'+experimentTag)
        plt.show




if WGradEpoch:
    # Average W gradient per epoch on logarithmic scale
    plt.figure(8)
    plt.title("Logarithm of the Norm of the Average W Gradient per Epoch")
    plt.plot(np.log(WGradEpoch),'b',label='Average W Gradient per Epoch')
    plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
    plt.xlabel('Epoch')
    plt.ylabel('Natural Logarithm Value')
    if HPC:
      plt.savefig(HPCDirectory+'WGrads'+experimentTag)  
    else:  
        plt.savefig('WGrads'+experimentTag)
        plt.show



# Influence function graph 

# Calculate min and max of all linearly transformed signals  
noisyDerivs=np.transpose(np.asarray([np.convolve(W,y[:,sample],'same') for sample in iteratorSamples]))
minVal=int(np.min(noisyDerivs))
maxVal=int(np.max(noisyDerivs))

#Create linear function with range min and max of all linearly transformed signals  
rangeArray=np.linspace(minVal,maxVal,signalLength)
influence=np.dot(np.asarray([np.asarray([np.exp((-(rangeArray[i]-iteratorMu[j])**2.)/(2.*sigma**2.)) for j in iteratorAlpha]).astype(np.float64) for i in iteratorN])
,alpha)

# Influence Function applied to array of range between min and max values of derivatives
plt.figure(9)
plt.title("Effect of Influence Function on Linear Signal")
plt.plot(rangeArray,'b',label='Linear Signal')
plt.plot(influence,'r',label='Linear Signal Applied to Influence Function')
plt.legend(bbox_to_anchor=(.38, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'influenceFunction'+experimentTag)  
else:  
    plt.savefig('influenceFunction'+experimentTag)
    plt.show



#Step Sizes for alpha and W per Epoch
plt.figure(10)
plt.title("Average Step Sizes per Epoch")
if learningRateAlphaEpoch:
    plt.plot(np.arange(len(learningRateAlphaEpoch))+1,learningRateAlphaEpoch,'b',label='Step Size Alpha')
if learningRateWEpoch:
    plt.plot(np.arange(len(learningRateWEpoch))+1,learningRateWEpoch,'r',label='Step Size W')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Epoch')
if HPC:
  plt.savefig(HPCDirectory+'stepSizes'+experimentTag)  
else:  
    plt.savefig('stepSizes'+experimentTag)
    plt.show




## Changes in function over sampled epochs graph 

# Calculate value of predicted functions over sampled epochs 
if alphaEpoch and WEpoch:
    epochEstimatesGraph=np.array([1,len(alphaEpoch)-int(len(alphaEpoch)/2),len(alphaEpoch)-int(len(alphaEpoch)/3),len(alphaEpoch)-1])
    pastEstimates=np.transpose(np.asarray([estimateSignal(WEpoch[epoch],y[:,exemplarySignal],alphaEpoch[epoch],C,stepSize,iteratorMu,sigma) for epoch in epochEstimatesGraph]).astype(np.float))
elif alphaEpoch:
    epochEstimatesGraph=np.array([1,len(alphaEpoch)-int(len(alphaEpoch)/2),len(alphaEpoch)-int(len(alphaEpoch)/3),len(alphaEpoch)-1])
    pastEstimates=np.transpose(np.asarray([estimateSignal(W,y[:,exemplarySignal],alphaEpoch[epoch],C,stepSize,iteratorMu,sigma) for epoch in epochEstimatesGraph]).astype(np.float))
elif WEpoch:
    epochEstimatesGraph=np.array([1,len(WEpoch)-int(len(WEpoch)/2),len(WEpoch)-int(len(WEpoch)/3),len(WEpoch)-1])
    pastEstimates=np.transpose(np.asarray([estimateSignal(WEpoch[epoch],y[:,exemplarySignal],alpha,C,stepSize,iteratorMu,sigma) for epoch in epochEstimatesGraph]).astype(np.float))



# Plot changes in function graph 
plt.figure(11)
plt.title("Change by Epoch of Predicted Train Signal: "+str(exemplarySignal+1))
plt.plot(iteratorN+1,x[:,exemplarySignal],'b',label='Original Signal')
for i in range(len(epochEstimatesGraph)):
    plt.plot(iteratorN+1,pastEstimates[:,i],label='Epoch: '+str(epochEstimatesGraph[i]+1))
plt.xlabel('Entry of Signal')
plt.legend(bbox_to_anchor=(.8, 1), loc=2, borderaxespad=0.)
if HPC:
  plt.savefig(HPCDirectory+'predChange'+experimentTag)  
else:  
    plt.savefig('predChange'+experimentTag)
    plt.show

           

      
# Graph mean SSR per sample for train and test epochs
plt.figure(12)
plt.plot(SSREpochAvg[4:],'r',label='Train SSR')
plt.plot(SSRTest[4:],'b',label='Test SSR')
plt.title("Train and Test SSR Sample Average per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Mean Sum of Squared Residual')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
if HPC:
  plt.savefig(HPCDirectory+'SSR'+experimentTag)  
else:  
    plt.savefig('meanSSR'+experimentTag)
    plt.show
    
    
os.system('say "free will is an illusion"')