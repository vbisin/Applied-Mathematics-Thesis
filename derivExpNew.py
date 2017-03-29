#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:15:57 2017

@author: vittoriobisin
"""
import numpy as np

def expFunctionNew(sampleY,l,W,i,alpha):
    alphaDim=len(alpha)
    sigma=30.
    #mu=N/alphaDim
    iteratorMu=np.zeros(alphaDim)
    iteratorMakeMu=np.arange(alphaDim/2)
    iteratorMakeMuRev=iteratorMakeMu[::-1]*(-30)-30

    iteratorMu[iteratorMakeMu]=iteratorMakeMuRev[iteratorMakeMu]
    iteratorMu[iteratorMakeMu+alphaDim/2+1]=iteratorMakeMu*30+30
    
                           
    entry=np.zeros(alphaDim)
    
    signal=np.dot(W[i,:],sampleY)
    iterator=np.arange(alphaDim)
    
    
    #entry[:]=np.asarray([np.exp((-(signal-center)**2)/(2*sigma**2))*(-sampleY[l]*alpha[center]*(signal-iteratorMu[center])/(sigma**2))  for center in iteratorMu]).astype(np.float64)
        

    entry[iterator]=np.exp(-(signal-iteratorMu[iterator])**2./(2.*sigma**2.))*(-sampleY[l]*alpha[iterator]*(signal-iteratorMu[iterator])/(sigma**2.))
    
    entry=sum(entry)
    return entry
