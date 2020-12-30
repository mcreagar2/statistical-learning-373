## This program approximates the gradient of a function given an array of input values as the
## point at which to compute the gradient. The program also computes the error between the true
## gradient and the approximation and plots the errors for a visual description of the accuracy
## of the approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
    

def gradApprox(f, x0, deltax):
    #does a numerical approximation of the gradient of a given function w an array
    n = len(x0)
    grad = np.zeros(n)
    for i in range(n):
        ##here, it is very important (!!) to use the .copy function. If one does
        ##not, x0up, x0down, and x0 all point to the same location in memory, which
        ##will cause the gradApprox function to return an array of zeroes.
        x0up = x0.copy()
        x0up[i] = x0[i] + deltax
        x0down = x0.copy()
        x0down[i] = x0[i] - deltax
        grad[i] = (f(x0up) - f(x0down))/(2.0*deltax)

    return grad   

def gradlogSum(x0):
    #takes an array as input and calculates the exact gradient of the function given in class
    n = len(x0)
    gradReal = np.zeros(n)
    denom = 1.0
    for j in range(n):
        denom = denom + np.exp(x0[j])
        
    for i in range(n):

        gradReal[i] = np.exp(x0[i])/(denom)
        
    return gradReal

def logSumExp(x):
    n = x.shape[0]
    s = 1.0
    for i in range(n):
        s = s + np.exp(x[i])
    return np.log(s)

x = 10*np.random.randn(5)
grad = gradlogSum(x)
dxVect = [0.1, 0.01, 0.001, 0.0001, 0.00001]
errors = np.zeros(5)
for j in range(5):
    gradEst = gradApprox(logSumExp,x,dxVect[j])
    nrm = np.linalg.norm(grad - gradEst)
    print(nrm)
    errors[j] = nrm
    

plt.plot(dxVect, errors)

##The error decreases quite dramatically with smaller delta values. However,
##delta values get too small, we get round-off error in the norm of the approx
##and the true gradient values. This causes an apparent increase in the
##difference as delta approaches 0.
