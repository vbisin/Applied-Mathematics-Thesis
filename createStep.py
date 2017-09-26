'''Creates random piece-wise constant functions '''

import numpy as np
import random 

def createSteps(numberExamples,signalLength,maxValue,noiseStdDev,minJumpSize):

## Initialize iterator arrays
    iteratorSamples=np.arange(numberExamples)
    iteratorLength=np.arange(signalLength)        
    
## Initialize x and y samples 
    
    #x is the clean signal  
    x=np.zeros((signalLength,numberExamples))
    
    #y is the noisy signal 
    y=np.zeros((signalLength,numberExamples))
 
   
## Function calling loop which defines each signal   
    
    x=np.transpose(np.asarray([speedCreateSteps(signalLength,maxValue,minJumpSize) for i in iteratorSamples]).astype(np.float))
       
    
## For each point in the original x signal, we add random Gaussian noise (0,noiseStd) to define y
    y[iteratorLength,:]= np.transpose([np.asarray([x[i,j]+int(np.random.normal(0,noiseStdDev)) for i in iteratorLength]).astype(np.float) for j in iteratorSamples])
    y[0,:]=x[0,:]
    
    return (x,y,numberOfPossibleJumps)   

def speedCreateSteps(length,maxValue,minJumpSize):

##Define number of jumps per signal
    global numberOfPossibleJumps

    # Number of possible jumps
    possibleJumps=[1,2,3]

    #Randomly pick one of the above number of possible jumps
    numberJumps=random.sample(possibleJumps,1)[0]
    numberOfPossibleJumps=len(possibleJumps)-1


    # Depending on the number of steps, we define a uniform "step" length for each signal
    intervalLength=length/numberJumps
    
    # Define signal to be returned 
    vectorX=np.zeros(length)


## For each step, we assign a random value between 0 and maxValue 
# We also fix the minimum Size of Jumps so that the difference in real signal cannot be less than
# minJumpSize
    
    for j in range(numberJumps):
        start=j*intervalLength
        end=(j+1)*intervalLength
        if j!=0: 
            vectorX[start:end]=int(random.uniform(0,maxValue))
            while abs(vectorX[start-1]-vectorX[start])<minJumpSize:
                vectorX[start:end]=int(random.uniform(0,maxValue))

        else:
            vectorX[start:end]=int(random.uniform(minJumpSize,maxValue))



    return (vectorX)