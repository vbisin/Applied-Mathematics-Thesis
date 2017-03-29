#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:36:58 2017

@author: vittoriobisin
"""
import numpy as np
#import time

#@jit
#def makedOmegadW(alphaDim,sampleX,sampleY,alpha,W):
#    #import matplotlib.pyplot as plt
#    # Where signal is a vector
#    
#    start_time = time.time()
#
#    N=len(sampleY)
#    mu=2
#    sigma=1
#    dOmegadW=np.zeros((N,N,N))
#    
#    for k in range(0,N):
#        for l in range(0,N):
#            for i in range(0,N):
#                for j in range(0,alphaDim):
#                    mu=mu*j
#
#    
#                    temp=np.exp((-(W[k,l]*sampleY[i]-mu)**2)/(2*sigma**2))
#                    temp=-temp*alpha[j]*sampleY[i]*(W[k,l]*sampleY[i]-mu)/sigma**2
#               
#                dOmegadW[i,l,k]=temp
#        print("1 k down")
#                
#    print "My program took", time.time() - start_time, "to run"
#                
#    return dOmegadW


def makedOmegadW(sampleY,alpha,W):
    from derivExpNew import expFunctionNew
    #start_time = time.time()
    N=len(sampleY)
    dOmegadW=np.zeros((N,N,N))
    iterator=np.arange(N)
    
    
    for i in iterator:
        lvector=np.array([expFunctionNew(sampleY,l,W,i,alpha) for l in iterator]).astype(np.float)    
        dOmegadW[i,i,:]=lvector
            
    #print "My program took", time.time() - start_time, "to run"
    return dOmegadW
    





