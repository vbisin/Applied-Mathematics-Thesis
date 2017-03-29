#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:00:25 2017

@author: vittoriobisin
"""
import numpy as np
from rbf import rbfF
from numpy import dot as dot
from numpy import transpose as trans
from makedOmegadW import makedOmegadW

def gammaDeriv(sampleX,sampleY,alpha,W):

    alphaDim=len(alpha)
    rbfWY=rbfF(dot(W,sampleY),alphaDim)
    
    # Where gamma is the original loss function
    gamma=dot(rbfWY,alpha)
    gamma=dot(trans(W),gamma)
    gamma=sampleX-(sampleY+gamma)
    
    # outputs a vector    
    return gamma

def alphaGradient(sampleX,sampleY,alpha,W):
  
    
    dFdG=gammaDeriv(sampleX,sampleY,alpha,W)
    N=len(sampleX)    
    alphaDim=len(alpha)
    
    dGdA=np.zeros((N,alphaDim))
    rbfWY=rbfF(dot(W,sampleY),alphaDim)
    negWTrans=-trans(W)
    
    iteratorN=np.arange(N)
    iteratorAlpha=np.arange(alphaDim)
    
    for i in iteratorN:
        dGdA[i,:]=np.asarray([dot(negWTrans[i,:],(rbfWY[:,j])) for j in iteratorAlpha]).astype(np.float) 

    # I.e. switch up dot product order to preserve the dimensions 
    
    dFdA=dot(trans(dGdA),dFdG) 
    return dFdA


def wGradient(sampleX,sampleY,alpha,W):
    
    dFdG=gammaDeriv(sampleX,sampleY,alpha,W)
    N=len(sampleX)
    alphaDim=len(alpha)
    negTransW=-trans(W)
    
    rbfWY=rbfF(dot(W,sampleY),alphaDim)
    omega=dot(rbfWY,alpha)
    
    LHSchainRule=np.zeros((N,N,N))
    iterator=np.arange(N)
    LHSchainRule[iterator,iterator,iterator]=-omega[iterator]
    
    
    dOmegadW=makedOmegadW(sampleY,alpha,W)
    #print("computing tensor dot product")
    RHSchainRule=dot(negTransW,dOmegadW)
    #print("computed tensor dot product")
    # I.e. switch up dot product order to preserve the dimensions 
    
    dFdW=dot(trans(LHSchainRule+RHSchainRule),dFdG)
    return dFdW