
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:43:31 2017

@author: vittoriobisin
"""

def multiSGDthres(x,y,alphaDim): 
    import numpy as np
    #from rbfNew import rbfF
    #from numpy import dot as dot
    from gradients import alphaGradient, wGradient
    from functionEval import functionEval
    #from backtrackArmijo import backtrackArmijoAlpha, backtrackArmijoW
    #initialize W matrix 
    N=x.shape[0]
    samples=x.shape[1]
    factsToReturn=list()



    #W = np.ones((N,N))*.001
    #W = np.ones((N,N))*.001
    #W=np.random.uniform(low=-1, high=1, size=(N,N))
#    W=np.zeros((N,N))
#    np.fill_diagonal(W,2)
#    for i in range(N-1):
#        W[i+1,i]=-1
#        W[i,i+1]=-1
    
   # W=np.ones((N,N))*.000001
    W=np.zeros((N,N))
    for i in range(N):
        if i==0:
            W[i,i]=1
        else:
            np.fill_diagonal(W,1)
            W[i,i-1]=-1
       
            
    
    alpha=np.ones(alphaDim)*.75
    
    #alpha=np.random.uniform(low=-.5, high=.5, size=(alphaDim,))

#    for i in range(400):
#        if i%2==0:
#            alpha[i]=1
#        else:
#            alpha[i]=-1


    errorEpoch=list()
    #Initialize error rates
    errorEpoch.append(np.mean(np.mean((x-y)**2,axis=0)))
    savedAlphaGrads=list()

    #stepSizeAlpha=list()
    #stepSizeW=list()

    maxSamples=np.arange(samples)
    #learningRateAlpha=.001
    learningRateAlpha=.0001
    #preLearningRateAlpha=1

    #learningRateW=.0001
    learningRateW=.000001           
    #preLearningRateW=1
    
    
    ## Different learning rates for alpha and w???
    alphaHistory=list()
    WHistory=list()
    alphaHistory.append(alpha)
    WHistory.append(W)
    
    #errorSample.append(np.sum(y[:,j]-[function evaluated with params and current x])
    factsToReturn.append(W[0,0])
    factsToReturn.append(alpha[0])
    factsToReturn.append(learningRateAlpha)
    factsToReturn.append(learningRateW)


    while len(errorEpoch)<2 or abs(errorEpoch[len(errorEpoch)-1]-errorEpoch[len(errorEpoch)-2])>1:
        errorSample=list()
        for j in maxSamples:
        #for j in range(10):

            sampleX=x[:,j]
            sampleY=y[:,j]
            #RHS of original equation 
            
            alphaGrad=alphaGradient(sampleX,sampleY,alpha,W)
            savedAlphaGrads.append(alphaGrad)
            #WGrad=wGradient(sampleX,sampleY,alpha,W)
            
            #learningRateAlpha=backtrackArmijoAlpha(W,sampleY,alpha,alphaGrad)
            #learningRateW=backtrackArmijoW(W,sampleY,alpha,W)
            
            #preLearningRateAlpha=learningRateAlpha
            #preLearningRateW=learningRateW
            #stepSizeAlpha.append(preLearningRateAlpha)
            #stepSizeW.append(preLearningRateW)

            
            alpha=alpha-learningRateAlpha*alphaGrad
            #W=W-learningRateW*WGrad
            
            estimate=functionEval(W,sampleY,alpha)
            errorSample.append(np.mean((sampleX-(sampleY+estimate))**2))
            
            alphaHistory.append(alpha)
            WHistory.append(W)
            
        errorEpoch.append(np.average(errorSample))
        print("Threshold SGD " + str(len(errorEpoch)-1))
        if len(errorEpoch)>2 and errorEpoch[len(errorEpoch)-1]-errorEpoch[len(errorEpoch)-2]>1000:
            print("diverged")
            break
                
    return (alpha, W, errorEpoch,errorSample,alphaHistory,WHistory,factsToReturn,savedAlphaGrads)            
            
                        
                        
 

                        
 
