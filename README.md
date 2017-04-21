# Learning Optimal Filter and Penalty Function for Denoising

Python code for a 1-D implementation of the denoising algorithm from "On learning optimized reaction diffusion processes for effective image restoration" by Yunjin Chen, Wei Yum, and Thomas Pock (https://arxiv.org/abs/1503.05768). 

## driver.py
The driver script for the algorithm. Outputs graphs and percent successfully denoised. 

##  createStep.py 
1. createSteps- Creates step (i.e. piece-wise constant) signals, where each function has a number of jumps equal to a multiple of the original signal's length. Gaussian noise is then added to each signal. 
2. speedCreateSteps - Creates step function (without noise) for each sample
## sgd.py
1. multiSGDthres - Runs stochastic gradient descent to minimize the the kernel (W) and radial basis function parameter (alpha). 
2. samplesSGDLoop - Computes alpha, W gradients, and error per sample 

## rbf.py
1. rbfF - Calculates the Gaussian functions in the Gaussian Radial Basis Function (before being multiplied by the minimized parameter, alpha).
2. GRBFCenters- Optimizes the equally spaced centers of the Gaussian RBF's given the input signal's range.  

## estimateSignal.py
Estimates values of predicted signal given current parameters.

## gradients.py
1. dFdGamma - calculates gradient of loss function w.r.t. variables inside the norm (needed for chain rule).
2. alphaGradient - calculates gradient of loss function w.r.t. alpha.
3. WGradient - calculates gradient of loss function w.r.t. the kernel (W).

## dOmegadW.py
1. dOmegadW - Returns derivative of omega (defined as the Gaussian RBF) w.r.t the kernel (W).
2. dOmegadWEntry - returns an entry of the above 3-tensor, dOmegadW.


