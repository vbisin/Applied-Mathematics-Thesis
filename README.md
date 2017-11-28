# Learning Optimal Filter and Penalty Function for Denoising

## driver.py
The driver script for the algorithm. Outputs graphs and percent successfully denoised for learned model on train and test sets. 

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
1. rbfF - Calculates the Gaussian Radial Basis Functions (before being multiplied by the minimized parameter, alpha).

## rbfCenters.py                                                                                  
1. GRBFCenters- Calculates the equally spaced centers of the Gaussian RBF's, given the ranges of the noisy signals under the linear transform W.  

## estimateSignal.py
Computes the model's predicted signal.

## gradients.py
1. dFdGamma - calculates gradient of objective function w.r.t. variables inside the norm.
2. alphaGradient - calculates gradient of objective function w.r.t. alpha.
3. WGradient - calculates gradient of objective function w.r.t. the kernel (W).

## dGammadW.py
1. dGammadW - Returns derivative of gamma (see gradients appendix in writeup) w.r.t the kernel (W).
2. dGammadWEntry - returns an entry of the above, dGammadW.

## Armijo.py
1. armijoAlpha - Estimates the optimal step size for the alpha gradient using the Armijo Rule. 
2. armijoW - Estimates the optimal step size for the W gradient using the Armijo Rule. 

## gradientCheck.py
1. alphaGradCheck - Checks if the gradient wrt alpha is correct.
2. speedAlphaGradCheck - Applies the limit derivative definition for each entry in alpha.
2. WGradCheck - Checks if the gradient wrt W is correct.
3. speedWGradCheck - Applies the limit derivative definition for each entry in the kernel, W.

## exponentGradCheck.py
1. exponentGradCheck - Checks the dGamma/dW derivative referred to in the latex (involved in computing the W gradient)
2. speedExponentGradCheck - Applies the limit derivative definition to each entry of this derivative. 
