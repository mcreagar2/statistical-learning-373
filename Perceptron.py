#Author: Molly Creagar
#coding the Perceptron algorithm (linear classifier) from scratch


import numpy as np
import matplotlib.pyplot as plt


def perceptron(Xpos,Xneg,t):
    #run algorithm 50000 times
    numEpochs = 50000
    #we want to see the boundary every 500 epochs
    boundaryVis = 500
    #starting vector
    a = np.random.randn(3)

    for i in range(numEpochs):       
        # random choice of epoch w/o replacement 
        for j in np.random.permutation(numXi):
            xi = X[j,:]
            #if we are in the first half of the data (the positive xi's)
            if j < numPos:
                #if a.dot(xi)>0, no need to do anything
                if a.dot(xi) < 0:
                    a = a + t*xi
            #if we are in the second half of the X vec (the negative xi's)
            else:
                #if a.dot(xi) <0, we are set    
                if a.dot(xi) > 0:
                    a = a - t*xi
                    
        # show the updates every 500 times        
        if i % boundaryVis == 0:
            print("Epoch ", i)
            plt.gcf().clear()
            plt.scatter(Xpos[:,0],Xpos[:,1],c="purple")
            plt.scatter(Xneg[:,0],Xneg[:,1],c="green")
            #replot the points and line
            plotLine(a,xMin,xMax,yMin,yMax,)
            plt.axis("equal")
            plt.pause(.05)
            plt.show()
            
    return a
    



def plotLine(a,xMin,xMax,yMin,yMax):
    #get the blue line between the clouds
    xVals = np.linspace(xMin,xMax,100)
    yVals = (-a[0] * xVals - a[2])/a[1]
    idxs =             np.where((yVals >= yMin) & (yVals <= yMax))
    plt.plot(xVals[idxs],yVals[idxs])

#create data
numPos = 100
numNeg = 100

np.random.seed(14)
muPos = [1.0, 1.0]
covPos = np.array([[1.0,0.0],[0.0,1.0]])

muNeg = [-1.0, -1.0]
covNeg = np.array([[1.0,0.0],[0.0,1.0]])


Xpos = np.ones((numPos,3))
for i in range(numPos):
    Xpos[i,0:2] = np.random.multivariate_normal(muPos, covPos)


Xneg = np.ones((numNeg,3))
for i in range(numNeg):
    Xneg[i,0:2] = np.random.multivariate_normal(muNeg,covNeg)


X = np.concatenate((Xpos,Xneg))
numPos = Xpos.shape[0]
numXi = X.shape[0]


xMin = -3.0
xMax = 3.0
yMin = -3.0
yMax = 3.0


t = .000001
a = perceptron(Xpos,Xneg,t)

