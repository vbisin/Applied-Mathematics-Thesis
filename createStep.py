## Creates random piece-wise constant functions 
import numpy as np
import random 

def createSteps(samples,length,maxValue,noiseStd):
## Initialize x and y samples 
    
    #x is the clean signal  
    x=np.zeros((length,samples))
    
    #y is the noisy signal 
    y=np.zeros((length,samples))
    
 
    iteratorSamples=np.arange(samples)
    iteratorLength=np.arange(length)    
    
# Determine multiples of lenth of signal, to then define number of uniform "steps" of each step function
    multiples=list()
    iteratorMultiples=np.arange(length)+1
    
    for i in iteratorMultiples:
        if length%i==0:
            multiples.append(i)
    
    
## Function calling loop which defines each signal   
    
    x=np.transpose(np.asarray([speedCreateSteps(length,maxValue,multiples) for i in iteratorSamples]).astype(np.float))
       
    
   ## For each point in the original x signal, we add random Gaussian noise (0,noiseStd) to define y
    y[iteratorLength,:]= np.transpose([np.asarray([x[i,j]+np.random.normal(0,noiseStd) for i in iteratorLength]).astype(np.float) for j in iteratorSamples])

    return (x,y)   

def speedCreateSteps(length,maxValue,multiples):

 # For each signal randomly pick one of the above defined multiples, this will be the number of "steps" or changes in values there will be 
    numberSteps=random.sample(multiples,1)[0]
    
    # Depending on the number of steps, we define a uniform "step" length for each signal
    intervalLength=length/numberSteps
    
    # Define vectors to be returned 
    vectorX=np.zeros(length)


## For each step, we assign a random value between 0 and maxValue 
    
    for j in range(numberSteps):
        start=j*intervalLength
        end=(j+1)*intervalLength
        vectorX[start:end]=int(random.uniform(0,maxValue))



    return (vectorX)