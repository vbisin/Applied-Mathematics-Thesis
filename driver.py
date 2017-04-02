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



## Define starting parameters for algorithm 

# Max value in range of each signal
maxValue=100

#Number of randomly generated signals to train over
numberExamples=500

#Length of each signal
signalLength=100

# Number of Gaussians in Gaussian RBF 
alphaDim=11

#Standard deviation of added noise 
noiseStdDev=25

#Retrive randomly generated piecewise constant signals (x is original, y is noisy)
(x,y)=createSteps(numberExamples,signalLength,maxValue,noiseStdDev)

##W and alpha Initalization 
    
W=np.ones((signalLength,signalLength))*.00001
#W=np.zeros((N,N))
for i in range(signalLength):
    if i==0:
        W[i,i]=1
    else:
        np.fill_diagonal(W,1)
        W[i,i-1]=-1

alpha=np.ones(alphaDim)*.75
#Approximate optimized values of alpha when SGD on it
#alpha=np.array([3.86,8.08,12.05,9.9,3.47,.48,-2.54,-8.667,-10.63,-6.75,-2.4], dtype=np.float)              
                  


#Run the Stochastic Gradient Descent Algorithm
(alpha, W, errorEpoch,errorSample,alphaHistory,WHistory,learningRates,savedAlphaGrads,savedWGrads)=multiSGDthres(x,y,alpha,W)




## Graphing parameters 
epochsArray=np.arange(len(errorEpoch))
iteratorAlpha=np.arange(alphaDim)
iteratorN=np.arange(signalLength)
iteratorSamples=np.arange(numberExamples)
EPOCH=(len(epochsArray)-1)*numberExamples+1
iteratorEpochs=np.arange(EPOCH)
      
#Exemplary signal to graph 
signal=200
    
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
plt.plot(iteratorN,y[:,signal],'r',label='Noisy signal (sigma=25): '+str(signal+1))
plt.legend(bbox_to_anchor=(.55, .85), loc=0, borderaxespad=0.)
plt.title("Sample of Original and Noisy Signals ("+str(signal+1)+" of "+ str(numberExamples)+")")
plt.savefig('signals')
plt.show

# Calculates MSE over each epoch
plt.figure(2)
plt.plot(epochsArray,errorEpoch,'r')
plt.title("Mean MSE per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('MSE')
plt.show

# Final estimate of alphas
plt.figure(3)
plt.plot(iteratorAlpha+1,alpha,'b',label='Alphas')
plt.title("RBF Parameter Values: Alpha")
plt.xlabel('Alpha Number')
plt.ylabel('Alpha Value')
plt.savefig('alphas')
plt.show


# Heat map of learned kernel
plt.figure(4)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title("Heat Map of Kernel Matrix")
plt.savefig('kernelHeat')
plt.show()


#Calculate sum of function evaluation estimates w.r.t. epochs
estimateSum=np.zeros(EPOCH)
WSum=np.zeros(EPOCH)
alphaSum=np.zeros(EPOCH)
for i in range(EPOCH):
    estimateSum[i]=sum(np.abs(y[:,signal]+np.dot(np.transpose(WHistory[i]),np.dot(rbfF(np.dot(WHistory[i],y[:,signal]),alphaDim),alphaHistory[i])))) 
    WSum[i]=sum(sum(np.abs(WHistory[i])))
    alphaSum[i]=sum(sum(np.abs(alphaHistory[i])))

# Absolute Sums of predicted and actual signals 
plt.figure(5)
plt.plot(iteratorEpochs,estimateSum,'b',label='Estimated Sum: Signal '+str(signal+1))
plt.plot(iteratorEpochs,np.ones(EPOCH)*sum(np.abs(x[:,signal])),'r',label='Actual Sum: Signal '+str(signal+1))
plt.title("Sum of Estimates")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Estimate Sum')
plt.xlabel('Iteration')
plt.savefig('sumEstimate')
plt.show

# Absolute sums of W matrix w.r.t. epochs
plt.figure(6)
plt.plot(iteratorEpochs,WSum,'b',label='Sum of W')
plt.title("Sum of W")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
plt.savefig('Wsum')
plt.show


# Absolute sums of alpha w.r.t. epochs
plt.figure(7)
plt.plot(iteratorEpochs,alphaSum,'b',label='Sum of alpha')
plt.title("Sum of alpha")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('Sum')
plt.xlabel('Iteration')
plt.savefig('alphaSum')
plt.show



# Original, noisy, and predicted signals 
plt.figure(8)
plt.title("Original, Noisy and Predicted Values for Signal: " +str(signal+1))
plt.plot(iteratorN,x[:,signal],'r',label='Original Signal')
plt.plot(iteratorN,y[:,signal],'b',label='Noisy Signal (sigma=25)')
plt.plot(iteratorN,y[:,signal]+finalEstimates[:,signal],'y',label='Predicted Signal')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Iteration')
plt.savefig('estimates')
plt.show


os.system('say "free will is an illusion"')
print "My program took", time.time() - start_time, "to run"
