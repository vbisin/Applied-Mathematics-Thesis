#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:25:41 2017

@author: vittoriobisin
"""

def functionEval(W,sampleY,alpha):
    import numpy as np
    from rbf import rbfF
    alphaDim=len(alpha)
    estimate=np.dot(rbfF(np.dot(W,sampleY),alphaDim),alpha) 
    estimate=np.dot(np.transpose(W),estimate)
    return estimate
