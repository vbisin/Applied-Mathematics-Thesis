
## Creates random piece-wise constant functions 
import numpy as np
import random 

def createSteps(samples,length,maxValue,noiseStd):
## Initialize x and y samples 
    
    #x is the clean signal  
    x=np.zeros((length,samples))
    
    #y is the noisy signal 
    y=np.zeros((length,samples))
    
    
## Loop defining each signal    
    for i in range(samples): 
       
        # For each signal randomly decide how many "steps" or changes in values there will be 
        numberSteps=random.sample(set([1, 2, 4, 5,10,20,25,50]), 1)[0]
        # Depending on the number of steps, we define a uniform "step" length for each signal
        intervalLength=length/numberSteps

    ## For each step, we assign a random value between 0 and maxValue 
        for j in range(numberSteps):
            start=j*intervalLength
            end=(j+1)*intervalLength
            x[start:end,i]=random.uniform(0,maxValue)
    
    ## For each point in the original x signal, we add random Gaussian noise (0,noiseStd) to 
     # define y
        for i in range(length):
            y[i,:]= np.asarray([x[i,j]+np.random.normal(0,noiseStd) for j in range(samples)]).astype(np.float)


    return (x,y)



