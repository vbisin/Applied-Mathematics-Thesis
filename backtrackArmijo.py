#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:03:28 2017

@author: vittoriobisin
"""
from estimateSignal import estimateSignal

def backtrackArmijoAlpha (W,sampleY,alpha,alphaGrad):
    stepSize=1
    tao=.5   
    counter=0
    
    xk1=estimateSignal(W,sampleY,alpha)
    xk2=estimateSignal(W,sampleY,alpha-stepSize*alphaGrad)
    RHS=xk1-.5*stepSize*sum(alphaGrad**2)
    while  not (xk2<=RHS).all():
        stepSize=stepSize*tao


        counter+=1
        print("Alpha "+str(counter))
    return stepSize

