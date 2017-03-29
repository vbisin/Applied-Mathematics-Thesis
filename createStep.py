#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:24:58 2017

@author: vittoriobisin
"""

def createSteps(samples,length,maxValue,noiseStd):
    import numpy as np
    import random 
    #import matplotlib.pyplot as plt

    x=np.zeros((length,samples))
    y=np.zeros((length,samples))
    
    for i in range(samples): 
        #numberSteps=random.sample(set([1, 2, 3, 4, 5,8,10,20,25,40,50,100,250,500,1000]), 1)[0]
       #modified for N=100
        numberSteps=random.sample(set([1, 2, 4, 5,10,20,25,50]), 1)[0]
        intervalLength=length/numberSteps

        for j in range(numberSteps):
            start=j*intervalLength
            end=(j+1)*intervalLength
            x[start:end,i]=random.uniform(0,maxValue)
        for i in range(length):
            y[i,:]= np.asarray([x[i,j]+np.random.normal(0,noiseStd) for j in range(samples)]).astype(np.float)

        
    #plt.plot(np.arange(length)+1,x[:,i])
    #plt.plot(np.arange(length)+1,x[:,i])
    #plt.plot(np.arange(length)+1,y[:,i])

    return (x,y)



