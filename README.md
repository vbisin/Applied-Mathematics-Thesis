# Learning Optimal Filter and Penalty Function for Denoising

Python code for a 1-D implementation of the denoising algorithm from "On learning optimized reaction diffusion processes for effective image restoration" by Yunjin Chen, Wei Yum, and Thomas Pock (https://arxiv.org/abs/1503.05768). 

## driver.py
The driver script for the algorithm. Outputs graphs and percent successfully denoised. 

##  createStep.py 
Creates step (i.e. piece-wise constant) signals, where each function has a number of jumps equal to a multiple of the original signal's length. At each one of these steps the function takes a random number between 1 and 100. A large amount of Gaussian noise (mean 0 and standard deviation 25) is then added to each signal. 

## sgd.py
Runs a stochastic gradient descent to minimize the the kernel (W) and radial basis function parameter (alpha). 

## rbf.py
Calculates the Gaussian functions in the Gaussian Radial Basis Function (before being multiplied by the minimized parameter, alpha).

## estimateSignal.py
Estimates values of predicted signal given current parameters.

## gradients.py
1. dFdGamma - calculates gradient of loss function w.r.t. variables inside the norm (needed for chain rule).
2. alphaGradient - calculates gradient of loss function w.r.t. alpha.
3. wGradient - calculates gradient of loss function w.r.t. the kernel (W).

## dOmegadW.py
1. dOmegadW - Returns derivative of omega (defined as the Gaussian RBF) w.r.t the kernel (W).
2. dOmegadWEntry - returns an entry of the above 3-tensor, dOmegadW.


