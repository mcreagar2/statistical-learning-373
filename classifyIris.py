%Classification of the Iris dataset (available on Kaggle)

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import os

os.chdir("/Users/Molly/Documents/MATH373")

import loadData

def crossEntropy(y, obs):
    return - np.sum((y) * np.log(abs(obs)))
    
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
            grad[:,k] = xi.T*(probs[k] - yi[k])
    return grad

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
            
def multilogReg(t, X, Y):
    numEx, numFeat = X.shape
    numEx, numClass = Y.shape
    maxIter = 50000
    showTrigger = 10000
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



X, Y = loadData.load_iris()


X_test, X_train, Y_test, Y_train = sk.model_selection.train_test_split(X, Y, test_size = 0.5)

beta, costs = multilogReg(0.00001, X_train, Y_train)
yhat = np.matmul(X_test, beta)


truth = np.argmax(Y_test,axis = 1)
Predictions = np.argmax(yhat, axis = 1)
wrong = np.where(Predictions != truth)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrong = k/tot
fracRight = 1 - fracWrong

print("Fraction Classified Incorrectly: ", fracWrong, \
      " Fraction Classified Correctly: ", fracRight )
n=len(costs)
idx = np.zeros(n)
idx = [(i + 1) for i in range(n)]
plt.figure()
plt.semilogy(idx, costs)
