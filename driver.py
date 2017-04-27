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
randomInitialization=False

if HPC:
    plt.switch_backend('agg')
    
## Define starting parameters for algorithm 

# Max value in range of each signal
maxValue=100

#Number of randomly generated signals to train over
numberExamples=500

#Length of each signal
signalLength=30

# Number of Gaussians in Gaussian RBF 
alphaDim=11

#Standard deviation of added noise 
noiseStdDev=25


#Retrive randomly generated piecewise constant signals (x is original, y is noisy)
(x,y)=createSteps(numberExamples,signalLength,maxValue,noiseStdDev)

##W and alpha Initalization 
W=np.zeros((signalLength,signalLength))

if randomInitialization:
    # Random initialization 
    print("W randomly initialized")  
    for i in range(signalLength):
        W[i,:]=np.asarray([random.uniform(0, 10) for j in range(signalLength)]).astype(np.float)       
    # Random initialization 
    print("alpha randomly initialized")         
    alpha=np.asarray([random.uniform(0, 10) for j in range(alphaDim)]).astype(np.float)        
else: 
    # Finite difference operator    
    print("W well initialized")
    for i in range(signalLength):
        if i==0:
            W[i,i]=1
        else:
            np.fill_diagonal(W,1)
            W[i,i-1]=-1
    # Constant initialization 
    print("alpha constantly initialized")
    alpha=np.ones(alphaDim)*.75

            
# Function initializing derivative of W transpose w.r.t W, so only need to compute once
negWTransDeriv=np.zeros((signalLength,signalLength,signalLength,signalLength))
for k in range(signalLength):
   for l in range(signalLength):
       negWTransDeriv[l,k,k,l]=-1 
            
      
#Run the Stochastic Gradient Descent Algorithm
(alpha, W, errorEpoch,alphaHistory,WHistory,alphaGradEpoch,WGradEpoch,learningRatesAlpha,learningRatesW)=multiSGDthres(x,y,alpha,W,negWTransDeriv)

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
    print("Percent denoised: "+str(100*(1-(sum(sum(np.abs(x-y-finalEstimates)))/sum(sum(np.abs(x-y)))))))
else: 
    print("denoising did not occur")

## Graphs

HPCDirectory="/home/vb704/Learning-Optimal-Filter-and-Penalty-Function/kernelHeat"

#Tag describing type of experiment undergone, to better name the graph 
experimentTag='.'+str(noiseStdDev)+'.'+str(randomInitialization)+'.'+str(numberExamples)

# Original and Noisy Signal
plt.figure(1)
plt.plot(iteratorN+1,x[:,signal],'b',label='Original signal: '+str(signal+1))
plt.plot(iteratorN+1,y[:,signal],'r',label='Noisy signal (sigma='+str(noiseStdDev)+'): '+str(signal+1))
plt.legend(bbox_to_anchor=(.55, .85), loc=0, borderaxespad=0.)
plt.title("Original and Noisy Signals ("+str(signal+1)+" of "+ str(numberExamples)+")")
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'signals'+experimentTag)  
else:    
    plt.savefig('signals'+experimentTag)
    plt.show

# Calculates MSE over each epoch
plt.figure(2)
plt.plot(epochsArray,errorEpoch,'r')
plt.title("MSE per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
if HPC:
  plt.savefig(HPCDirectory+'MSE'+experimentTag)  
else:  
    plt.savefig('MSE'+experimentTag)
    plt.show

# Final estimate of alphas
plt.figure(3)
plt.plot(iteratorAlpha+1,alpha,'b',label='Alphas')
plt.title("RBF Parameter Values: Alpha")
plt.xlabel('Alpha Number')
plt.ylabel('Alpha Value')
if HPC:
  plt.savefig(HPCDirectory+'alphas'+experimentTag)  
else:  
    plt.savefig('alphas'+experimentTag)
    plt.show



# Heat map of learned kernel (W)
plt.figure(4)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Heat Map of Kernel Matrix")
if HPC:
  plt.savefig(HPCDirectory+'kernelHeat'+experimentTag)  
else:  
    plt.savefig('kernelHeat'+experimentTag)
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
plt.plot(iteratorEpochs+1,estimateSum,'b',label='Estimated Sum of Absolute Value of Signal '+str(signal+1))
plt.plot(iteratorEpochs+1,np.ones(EPOCH)*sum(np.abs(x[:,signal])),'r',label='Actual Sum: Signal '+str(signal+1))
plt.title("Sum of Estimates")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Estimate Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig(HPCDirectory+'sumEstimate'+experimentTag)  
else:  
    plt.savefig('sumEstimate'+experimentTag)
    plt.show()


WSum[iteratorEpochs]=np.sum(np.sum(np.abs(WHistory[iteratorEpochs]),axis=1),axis=1)
# Absolute sums of W matrix w.r.t. epochs
plt.figure(6)
plt.plot(iteratorEpochs+1,WSum,'b',label='Sum of Absolute Value of W')
plt.title("Sum of Absolute values of W")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig(HPCDirectory+'Wsum'+experimentTag)  
else:  
    plt.savefig('Wsum'+experimentTag)
    plt.show()


# Absolute sums of alpha w.r.t. epochs
plt.figure(7)
plt.plot(iteratorEpochs+1,alphaSum,'b',label='Sum of Absolute Value of alpha')
plt.title("Sum of Absolute Value of alpha")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
if HPC:
  plt.savefig(HPCDirectory+'alphaSum'+experimentTag)  
else:  
    plt.savefig('alphaSum'+experimentTag)
    plt.show()


# Original, noisy, and predicted signals 
plt.figure(8)
plt.title("Original, Noisy, and Predicted Values for Signal: " +str(signal+1))
plt.plot(iteratorN+1,x[:,signal],'b',label='Original Signal')
plt.plot(iteratorN+1,y[:,signal],'r',label='Noisy Signal (sigma='+str(noiseStdDev)+')')
plt.plot(iteratorN+1,y[:,signal]+finalEstimates[:,signal],'k',label='Predicted Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'estimates'+experimentTag)  
else:  
    plt.savefig('estimates'+experimentTag)
    plt.show

gradIterator=np.arange(len(alphaGradEpoch))

# Average Alpha gradient per epoch
plt.figure(9)
plt.title("Average Alpha Gradient per Epoch")
plt.plot(gradIterator+1,alphaGradEpoch,'b',label='Average Alpha Gradient per Epoch')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Epoch')
if HPC:
  plt.savefig(HPCDirectory+'AlphaGrads'+experimentTag)  
else:  
    plt.savefig('AlphaGrads'+experimentTag)
    plt.show

# Average W gradient per epoch
plt.figure(10)
plt.title("Average W Gradient per Epoch")
plt.plot(gradIterator+1,WGradEpoch,'b',label='Average W Gradient per Epoch')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Epoch')
if HPC:
  plt.savefig(HPCDirectory+'WGrads'+experimentTag)  
else:  
    plt.savefig('WGrads'+experimentTag)
    plt.show

influence=np.dot(rbfF(np.range(len(y[:,0]))),alpha)

# Influence Function, applied to incremental ticker array
plt.figure(11)
plt.title("Influence Function")
plt.plot(np.arange(len(y[:,signal]))+1,influence,'b',label='Influence Function applied to incremental array from 0 to '+str(len(y[:,0])))
#plt.plot(np.arange(len(y[:,signal])),np.dot(W,y[:,signal]),'r',label='Original Convolved Image for Signal '+str(signal))
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Entry of Signal')
if HPC:
  plt.savefig(HPCDirectory+'influenceFunction'+experimentTag)  
else:  
    plt.savefig('influenceFunction'+experimentTag)
    plt.show


learningRateAlphasAvg=np.zeros(len(learningRatesAlpha)/numberExamples)
learningRateWAvg=np.zeros(len(learningRatesW)/numberExamples)
for i in range(len(learningRatesW)/numberExamples):
    start=i*numberExamples
    end=(i+1)*numberExamples
    learningRateAlphasAvg[i]=np.mean(np.array(learningRatesAlpha[start:end]))
    learningRateWAvg[i]=np.mean(np.array(learningRatesW[start:end]))


#Step Sizes for alpha and W per Epoch
plt.figure(12)
plt.title("Average Alpha and W Step Sizes per Epoch")
plt.plot(np.arange(len(learningRateAlphasAvg))+1,learningRateAlphasAvg,'b',label='Step Size Alpha')
plt.plot(np.arange(len(learningRateWAvg))+1,learningRateWAvg,'r',label='Step Size W')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Epoch')
if HPC:
  plt.savefig(HPCDirectory+'stepSizes'+experimentTag)  
else:  
    plt.savefig('stepSizes'+experimentTag)
    plt.show




os.system('say "free will is an illusion"')
print "My program took", time.time() - start_time, "to run"
