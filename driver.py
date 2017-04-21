## Driver which denoises a signal from randomly created step functions by learning a kernel and 
# parameters for its respective Gaussian Radial Basis Function

import numpy as np
from createStep import createSteps
import matplotlib.pyplot as plt
from rbf import rbfF
import time
import os
from sgd import multiSGDthres
from estimateSignal import estimateSignal
start_time = time.time()
import random

HPC=False

## Define starting parameters for algorithm 

# Max value in range of each signal
maxValue=100

#Number of randomly generated signals to train over
numberExamples=1000

#Length of each signal
signalLength=50

# Number of Gaussians in Gaussian RBF 
alphaDim=11

#Standard deviation of added noise 
noiseStdDev=5

# Step sizes/learning rates for Stochastic Gradient Descent
if noiseStdDev<25:
    learningRateAlpha=.0000005
    learningRateW=.0000005    
else: 
    learningRateAlpha=.000005
    learningRateW=.000005    
    

#Retrive randomly generated piecewise constant signals (x is original, y is noisy)
(x,y)=createSteps(numberExamples,signalLength,maxValue,noiseStdDev)

##W and alpha Initalization 
W=np.zeros((signalLength,signalLength))

# Random initialization 
#print("W randomly initialized")  
#for i in range(signalLength):
#    W[i,:]=np.asarray([random.uniform(0, 10) for j in range(signalLength)]).astype(np.float)
#    
# Finite difference operator    
print("W well initialized")
for i in range(signalLength):
    if i==0:
        W[i,i]=1
    else:
        np.fill_diagonal(W,1)
        W[i,i-1]=-1

# Random initialization 
#print("alpha randomly initialized")         
#alpha=np.asarray([random.uniform(0, 10) for j in range(alphaDim)]).astype(np.float)        

# Constant initialization 
print("alpha well initialized")
alpha=np.ones(alphaDim)*.75
             
# Function initializing derivative of W transpose w.r.t W, so only need to compute once
negWTransDeriv=np.zeros((signalLength,signalLength,signalLength,signalLength))
for k in range(signalLength):
   for l in range(signalLength):
       negWTransDeriv[l,k,k,l]=-1 
            
      
#Run the Stochastic Gradient Descent Algorithm
(alpha, W, errorEpoch,alphaHistory,WHistory,learningRates,alphaGradEpoch,WGradEpoch)=multiSGDthres(x,y,alpha,W,learningRateAlpha,learningRateW,negWTransDeriv)

# Used when running on HPC
#os.chdir('/home/vb704/Learning-Optimal-Filter-and-Penalty-Function')
      
#Exemplary signal to graph 
signal=0
    

## Graphing parameters 
epochsArray=np.arange(len(errorEpoch))
iteratorAlpha=np.arange(alphaDim)
iteratorN=np.arange(signalLength)
iteratorSamples=np.arange(numberExamples)
EPOCH=(len(epochsArray)-1)*numberExamples+1
iteratorEpochs=np.arange(EPOCH)

#Calculate final estimated signal for each signal (i.e. after the algorithm has converged)
finalEstimates=np.zeros((signalLength,numberExamples))
finalEstimates[:,iteratorSamples]=np.transpose(np.asarray([estimateSignal(W,y[:,j],alpha)-y[:,j] for j in iteratorSamples]).astype(np.float))


## Print statement if the algorithm denoised the signal (and calculate percent denoised)
if sum(sum(np.abs(x-y)))>sum(sum(np.abs(x-y-finalEstimates))):
    print ("Denoising did occur")
    print("Percent denoised: "+str((1-sum(sum(np.abs(x-y-finalEstimates)))/sum(sum(np.abs(x-y))))*100))
else: 
    print("denoising did not occur")





## Graphs

# Original and Noisy Signal
plt.figure(1)
plt.plot(iteratorN,x[:,signal],'b',label='Original signal: '+str(signal+1))
plt.plot(iteratorN,y[:,signal],'r',label='Noisy signal (sigma='+str(noiseStdDev)+'): '+str(signal+1))
plt.legend(bbox_to_anchor=(.55, .85), loc=0, borderaxespad=0.)
plt.title("Sample of Original and Noisy Signals ("+str(signal+1)+" of "+ str(numberExamples)+")")
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/signals")  
else:    
    plt.savefig('signals')
    plt.show

# Calculates MSE over each epoch
plt.figure(2)
plt.plot(epochsArray,errorEpoch,'r')
plt.title("MSE per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/MSE")  
else:  
    plt.savefig('MSE')
    plt.show

# Final estimate of alphas
plt.figure(3)
plt.plot(iteratorAlpha+1,alpha,'b',label='Alphas')
plt.title("RBF Parameter Values: Alpha")
plt.xlabel('Alpha Number')
plt.ylabel('Alpha Value')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/alphas")  
else:  
    plt.savefig('alphas')
    plt.show



# Heat map of learned kernel (W)
plt.figure(4)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title("Heat Map of Kernel Matrix")
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/kernelHeat")  
else:  
    plt.savefig('kernelHeat')
    plt.show()


#Calculate sum of function evaluation estimates w.r.t. epochs
estimateSum=np.zeros(EPOCH)
WSum=np.zeros(EPOCH)
alphaSum=np.zeros(EPOCH)
for i in iteratorEpochs:
    estimateSum[i]=sum(np.abs(y[:,signal]+np.dot(np.transpose(WHistory[i]),np.dot(rbfF(WHistory[i],y[:,signal],alphaDim),alphaHistory[i])))) 
alphaSum[iteratorEpochs]=np.sum(np.abs(alphaHistory[iteratorEpochs]),axis=1)

# Absolute Sums of predicted and actual signals 
plt.figure(5)
plt.plot(iteratorEpochs,estimateSum,'b',label='Estimated Sum of Absolute Value of Signal '+str(signal+1))
plt.plot(iteratorEpochs,np.ones(EPOCH)*sum(np.abs(x[:,signal])),'r',label='Actual Sum: Signal '+str(signal+1))
plt.title("Sum of Estimates")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Estimate Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/sumEstimate")  
else:  
    plt.savefig('sumEstimate')
    plt.show()


WSum[iteratorEpochs]=np.sum(np.sum(np.abs(WHistory[iteratorEpochs]),axis=1),axis=1)
# Absolute sums of W matrix w.r.t. epochs
plt.figure(6)
plt.plot(iteratorEpochs,WSum,'b',label='Sum of Absolute Value of W')
plt.title("Sum of Absolute values of W")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/Wsum")  
else:  
    plt.savefig('Wsum')
    plt.show()


# Absolute sums of alpha w.r.t. epochs
plt.figure(7)
plt.plot(iteratorEpochs,alphaSum,'b',label='Sum of Absolute Value of alpha')
plt.title("Sum of Absolute Value of alpha")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/alphaSum")  
else:  
    plt.savefig('alphaSum')
    plt.show()


# Original, noisy, and predicted signals 
plt.figure(8)
plt.title("Original, Noisy, and Predicted Values for Signal: " +str(signal+1))
plt.plot(iteratorN,x[:,signal],'b',label='Original Signal')
plt.plot(iteratorN,y[:,signal],'r',label='Noisy Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN,y[:,signal]+finalEstimates[:,signal],'k',label='Predicted Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/estimates")  
else:  
    plt.savefig('estimates')
    plt.show

gradIterator=np.arange(len(alphaGradEpoch))

plt.figure(9)
plt.title("Average sum of Alpha Gradient per Epoch")
plt.plot(gradIterator,alphaGradEpoch,'b',label='Average sum of Alpha Gradient per Epoch')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/alphaGradsSum")  
else:  
    plt.savefig('AlphaGradsSum')
    plt.show


if len(WGradEpoch)>1:
    plt.figure(10)
    plt.title("Average sum of W Gradient per Epoch")
    plt.plot(gradIterator,WGradEpoch,'b',label='Average sum of W Gradient per Epoch')
    plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
    plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/WGradsSum")  
else:  
    plt.savefig('WGradsSum')
    plt.show

influence=np.dot(rbfF(W,y[:,signal],len(alpha)),alpha)
plt.figure(11)
plt.title("Influence Function")
plt.plot(np.arange(len(y[:,signal])),influence,'b',label='Influence Function for Signal '+str(signal))
plt.plot(np.arange(len(y[:,signal])),np.dot(W,y[:,signal]),'r',label='Original Convolved Image for Signal '+str(signal))
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Iteration')
if HPC:
  plt.savefig("/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/influenceFunction")  
else:  
    plt.savefig('influenceFunction')
    plt.show




os.system('say "free will is an illusion"')
print "My program took", time.time() - start_time, "to run"
