#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:22:56 2017

@author: vittoriobisin
"""

# Let's define the original signal
import numpy as np
from createStep import createSteps
from SGD import multiSGD
import matplotlib.pyplot as plt
from rbfNew import rbfF
import time
from SGDthres import multiSGDthres
from functionEval import functionEval
start_time = time.time()

# Create these step functions
maxValue=100
samples=500
signalLength=100
alphaDim=11
#alphaDim=(maxValue+2*noiseStd)*2*10+1
noiseStd=25

#epochs=100


(x,y)=createSteps(samples,signalLength,maxValue,noiseStd)
#x=np.ones((signalLength,samples))*10
#y=np.zeros((signalLength,samples))
#for i in range(signalLength/2):
#    x[i,:]=0
#for i in range(signalLength):
 #   y[i,:]= np.asarray([x[i,j]+np.random.normal(0,1) for j in range(samples)]).astype(np.float)
      
      

#Two Different SGD's
(alpha, W, errorEpoch,errorSample,alphaHistory,WHistory,extraFacts,savedAlphaGrads)=multiSGDthres(x,y,alphaDim)
#(alpha, W, errorEpoch,errorSample,alphaHistory,WHistory,extraFacts)=multiSGD(x,y,alphaDim,epochs)

epochsArray=np.arange(len(errorEpoch))
iteratorAlpha=np.arange(alphaDim)
iteratorN=np.arange(signalLength)


i=496
plt.figure(1)
plt.plot(iteratorN,x[:,i],'b',label='Original signal: '+str(i+1))
plt.plot(iteratorN,y[:,i],'r',label='Noisy signal (sigma=25): '+str(i+1))
plt.legend(bbox_to_anchor=(.55, .85), loc=0, borderaxespad=0.)
plt.title("Sample of Original and Noisy Signals ("+str(i+1)+" of "+ str(samples)+")")
plt.savefig('signals')
plt.show


plt.figure(2)
plt.plot(epochsArray,errorEpoch,'r')
plt.title("Mean MSE per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('MSE')
plt.show

plt.figure(3)
plt.plot(iteratorAlpha+1,alpha,'b',label='Alphas')
plt.title("RBF Parameter Values: Alpha")
plt.xlabel('Alpha Number')
plt.ylabel('Alpha Value')
plt.savefig('alphas')
plt.show

plt.figure(4)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title("Heat Map of Kernel Matrix")
plt.savefig('kernelHeat')
plt.show()

#plt.figure(5)
#plt.plot(iteratorN,x[:,0],'b',label='Original signal: sample 1')
#plt.plot(iteratorN,np.dot(W,y[:,0]),'r',label='Effect of kernel on sample 1')
#plt.legend(bbox_to_anchor=(.48, .89), loc=0, borderaxespad=0.)
#plt.title("Kernel")
#plt.savefig('Kernel')
#plt.show
#

gaussiansMatrix1=rbfF(np.dot(W,y[:,0]),alphaDim)
rbfResult1=np.dot(gaussiansMatrix1,alpha)
gaussiansMatrix2=rbfF(np.dot(W,y[:,1]),alphaDim)
rbfResult2=np.dot(gaussiansMatrix2,alpha)

#rbfResult=np.dot(np.transpose(W),rbfResult)
finalEstimates=np.zeros((signalLength,samples))
iteratorSamples=np.arange(samples)
finalEstimates[:,iteratorSamples]=np.transpose(np.asarray([functionEval(W,y[:,j],alpha) for j in iteratorSamples]).astype(np.float))
    
finalEstimate1=finalEstimates[:,0]
finalEstimate2=finalEstimates[:,1]

#plt.figure(6)
##plt.plot(iteratorN,x[:,0],'k',label='Original signal: sample 1')
#plt.plot(iteratorN,np.dot(W,y[:,0]),'r',label='Effect of kernel on sample 1')
#plt.plot(iteratorN,rbfResult1,'b',label='Influence Function for sample1')
#plt.plot(iteratorN,finalEstimate1,'k',label='final Estimate  for sample 1')
#plt.legend(bbox_to_anchor=(.48, .89), loc=0, borderaxespad=0.)
#plt.title("Influence Function")
#plt.savefig('influenceFunction')
#plt.show


#plt.figure(7)
#plt.plot(iteratorN,W[:,0],'b')
#plt.title("First Column of W")
#plt.ylabel('W')
#plt.savefig('distribution W')
#plt.show

EPOCH=(len(epochsArray)-1)*samples+1
epochsIterator=np.arange(EPOCH)

#alphasums=np.zeros(EPOCH)
#Wsums=np.zeros(EPOCH)
#RBFsums=np.zeros(EPOCH)

#for i in range(EPOCH):
#    alphasums[i]=sum(alphaHistory[i])
#    Wsums[i]=sum(sum(WHistory[i]))
#    RBFsums[i]=sum(sum(rbfF(np.dot(WHistory[i],y[:,0]),alphaDim)))
##
#plt.figure(8)
#plt.plot((epochsIterator),alphasums,'b')
#plt.title("sum of Alphas")
#plt.ylabel('Alpha Sum')
#plt.xlabel('Iteration')
#plt.savefig('alphaSum')
#plt.show
#
#
#plt.figure(9)
#plt.plot((epochsIterator),Wsums,'b')
#plt.title("sum of W's")
#plt.ylabel('W Sum')
#plt.xlabel('Iteration')
#plt.savefig('WSum')
#plt.show
#
#
#
#plt.figure(10)
#plt.plot(epochsIterator,RBFsums,'b')
#plt.title("sum of RBF's")
#plt.ylabel('RBF Sum')
#plt.xlabel('Iteration')
#plt.savefig('RBFSum')
#plt.show

estimateSum=np.zeros(EPOCH)

for i in range(EPOCH):
    estimateSum[i]=sum(y[:,0]-np.dot(np.transpose(WHistory[i]),np.dot(rbfF(np.dot(WHistory[i],y[:,0]),alphaDim),alphaHistory[i]))) 
    

plt.figure(11)
plt.plot(epochsIterator,estimateSum,'b',label='estimated sum: signal 1')
plt.plot(epochsIterator,np.ones(EPOCH)*sum(x[:,0]),'r',label='actual sum: signal 1')
plt.title("sum of estimates")
plt.legend(bbox_to_anchor=(.24, .89), loc=0, borderaxespad=0.)
plt.ylabel('estimate Sum')
plt.xlabel('Iteration')
plt.savefig('sumEstimate')
plt.show


#
#plt.figure(12)
#plt.plot(iteratorN,x[:,0],'r',label='actual sample 1')
#plt.plot(iteratorN,finalEstimate1,'b',label='final Estimate  for sample 1')
#plt.plot(iteratorN,x[:,1],'k',label='actual sample 2')
#plt.plot(iteratorN,finalEstimate2,'g',label='final Estimate  for sample 2')
#plt.title("final estimate")
#plt.legend(bbox_to_anchor=(.32, .89), loc=0, borderaxespad=0.)
#plt.xlabel('Iteration')
#plt.savefig('estimates')
#plt.show



#for i in range(samples):
i=0
plt.figure(13)
plt.title("Original, Noisy and Predicted Values for Signal: " +str(i+1) )
plt.plot(iteratorN,x[:,i],'r',label='Original Signal')
plt.plot(iteratorN,y[:,i],'b',label='Noisy Signal (sigma=25)')
plt.plot(iteratorN,y[:,i]+finalEstimate1,'y',label='Predicted Signal')
#plt.plot(iteratorN,np.dot(rbfF(np.dot(W,y[:,0]),alphaDim),alpha) ,'k',label='pre W transpose')
plt.legend(bbox_to_anchor=(.52, .24), loc=0, borderaxespad=0.)
plt.xlabel('Iteration')
plt.savefig('estimates')
plt.show


if sum(sum(np.abs(x-y)))>sum(sum(np.abs(x-y-finalEstimates))):
    print ("oh, yeah")
print(sum(sum(np.abs(x-y))))
print(sum(sum(np.abs(x-y-finalEstimates))))

    


#rbfApprox=np.dot(rbfF(x[:,1],alphaDim),alpha)
#plt.figure(13)
#plt.plot(iteratorN,np.dot(W,y[:,1]),'b',label='Signal: W*y')
#plt.plot(iteratorN,rbfApprox,'r',label='Influence Function Approx')
#plt.title("Goodness of Fit - Influence Function")
##plt.savefig('estimate')
#plt.legend(bbox_to_anchor=(.32, .89), loc=0, borderaxespad=0.)
#plt.xlabel('Iteration')
#plt.show

import os
os.system('say "free will is an illusion"')

print "My program took", time.time() - start_time, "to run"
