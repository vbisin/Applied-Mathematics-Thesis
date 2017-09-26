# Learning Optimal Filter and Penalty Function for Denoising

Python code for a 1-D implementation of total variation denoising with a Gaussian Kernel non-linearity

## driver.py
The driver script for the algorithm. Outputs graphs and percent successfully denoised for train and test sets. 

##  createStep.py 
1. createSteps- Creates step (i.e. piece-wise constant) signals, where each function has a number of jumps equal to a multiple of the original signal's length. Gaussian noise is then added to each signal. 
2. speedCreateSteps - Creates step function (without noise) for each sample
                                            
## sgd.py
1. multiSGDthres - Runs the stochastic gradient descent to minimize the the kernel (W) and/or the radial basis function parameter (alpha). 
2. SGDSample - Updates alpha and W values after computing gradient per sample.

## batchSGD.py 
1. batchSGD - runs the mini-batch stochastic gradient descent algorithm to minimize the the kernel (W) and/or the radial basis function parameter (alpha). 
2. batchSGDsample - Updates alpha and W values after computing gradient per batch.

## rbf.py
1. rbfF - Calculates the Gaussian functions in the Gaussian Radial Basis Function (before being multiplied by the minimized parameter, alpha).

## rbfCenters.py                                                                                  
1. GRBFCenters- Calculates equally spaced centers of the Gaussian RBF's, given the ranges of the noisy signals under the linear transfors W.  

## estimateSignal.py
Computes the predicted finite differnces of the signal given its current parameters.

## gradients.py
1. dFdGamma - calculates gradient of loss function w.r.t. variables inside the norm (needed for chain rule).
2. alphaGradient - calculates gradient of loss function w.r.t. alpha.
3. WGradient - calculates gradient of loss function w.r.t. the kernel (W).

## dGammadW.py
1. dGammadW - Returns derivative of omega (defined as the Gaussian RBF) w.r.t the kernel (W).
2. dGammadWEntry - returns an entry of the above, dOmegadW.

## Armijo.py
1. armijoAlpha - Estimates the optimal step size for the alpha gradient using the Armijo Rule. 
2. armijoW - Estimates the optimal step size for the W gradient using the Armijo Rule. 

## gradientCheck.py
1. alphaGradCheck - Checks if the gradient wrt alpha is correct (applying the limit definition of a derivative). 
2. speedAlphaGradCheck - Applies the limit derivative definition for each entry in alpha.
2. WGradCheck - Checks if the gradient wrt W is correct (applying the limit definition of a derivative).
3. speedWGradCheck - Applies the limit derivative definition for each entry in the kernel, W.

## exponentGradCheck.py
1. exponentGradCheck - Checks the dOmega/dW derivative referred to in the latex (involved in computing the W gradient)
2. speedExponentGradCheck - Applies the limit derivative definition to each entry of this derivative. 
