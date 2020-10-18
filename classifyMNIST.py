#Author: Molly Creagar
#Using multi-logistic regression, classify images from MNIST dataset. 
#Multi-logistic regression written from scratch

import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

#cost function    
def evalF(beta, X, Y):
    numEx, numFeat = X.shape
    numEx, numClass = Y.shape
    dots = np.matmul(X, beta)
    numers = np.exp(dots)
    denoms = np.sum(numers, axis = 1)
    denoms = np.reshape(denoms, (numEx, 1))
    denoms = np.tile(denoms, (1, numClass))
    p = numers/denoms
    logp = np.log(p)
    Ylogp = Y*logp
    return -np.sum(Ylogp)
        
#calculating gradient from scratch        
def evalGrad(beta, X, Y):
    numEx, numFeat = X.shape
    numEx, numClass = Y.shape
    grad = np.zeros((numFeat, numClass))
    for i in range(numEx):
        xi = X[i,:]
        yi = Y[i,:]
        dots = xi.dot(beta)
        terms = np.exp(dots)
        probs = terms/np.sum(terms)
        for k in range(numClass):
            grad[:,k] = (probs[k] - yi[k])* xi
    return grad
    
#faster gradient using matrices
def fastGrad(beta, X, Y):
    numEx, numFeat = X.shape
    numEx, numClass = Y.shape
    dots = np.matmul(X, beta)
    numers = np.exp(dots)
    denoms = np.sum(numers, axis = 1)
    denoms = np.reshape(denoms, (numEx, 1))
    denoms = np.tile(denoms, (1, numClass))
    p = numers/denoms
    grad = np.matmul(X.T, (p-Y))
    return grad

#multilog function call
def multilogReg(t, X, Y):
    numEx, numFeat = X.shape
    numEx, numClass = Y.shape
    maxIter = 500
    #create maximum number of iterations and a print trigger
    showTrigger = 50
    costs = np.zeros(maxIter)
    beta=np.zeros((numFeat, numClass))
    for i in range(maxIter):
        grad = fastGrad(beta, X, Y)
        beta = beta - t * grad
        cost = evalF(beta, X, Y)
        costs[i] = cost
        if i % showTrigger == 0:
            print("Iteration ", i, "; Cost: ", cost)
    return (beta, costs)

#get training data
(X_train,Y_train), (X_test, Y_test) = mnist.load_data()



X_train = X_train.reshape((60000, 28*28))
X_train = X_train.astype('float32')/255

X_test_orig = X_test.copy()
X_test = X_test.reshape((10000, 28*28))
X_test = X_test.astype('float32')/255


Y_train = tensorflow.keras.utils.to_categorical(Y_train)
Y_test = tensorflow.keras.utils.to_categorical(Y_test)

#get results from training
beta, costs = multilogReg(0.00001, X_train, Y_train)

#calculate how correct it is
Yhat =np.matmul(X_test,beta)
truth = np.argmax(Y_test,axis = 1)
Predictions = np.argmax(Yhat, axis = 1)
wrong = np.where(Predictions != truth)[0]
fracWrong = len(wrong)/len(Y_test)
fracRight = 1 - len(wrong)/len(Y_test)

print("Fraction Classified Incorrectly: ", fracWrong, \
      " Fraction Classified Correctly: ", fracRight )
n=len(costs)
idx = np.zeros(n)
idx = [(i + 1) for i in range(n)]
plt.figure()
plt.semilogy(idx, costs)
