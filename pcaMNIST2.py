import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers


(X_train,Y_train), (X_test, Y_test) = mnist.load_data()



X_train = X_train.reshape((60000, 28*28))
X_train = X_train.astype('float32')/255

X_test_orig = X_test.copy()
X_test = X_test.reshape((10000, 28*28))
X_test = X_test.astype('float32')/255

X = X_train[0:20000, :]
Y_train = tensorflow.keras.utils.to_categorical(Y_train)
Y = Y_train[0:20000, :]
Y_test = tensorflow.keras.utils.to_categorical(Y_test)


(numRows, numCols) = X.shape

Y = np.zeros((numRows, numCols))
Xbar = np.mean(X, axis = 0)
Y = X - Xbar

#compute SVD
#svd returns V transpose
U, Sigma, VT = np.linalg.svd(Y)

V = VT.T

##sigma right now is a vector-- let's make it a matrix
S=np.zeros((numRows, 784))
S[0:784, 0:784] = np.diag(Sigma)

#check if we did it right

check = U @ S @ VT

diff = np.max(np.abs(Y-check))

#let's look at a random image
r = np.random.randint(0, 784)
vr = X[r,:]
vR = np.reshape(vr, (28,28))
plt.imshow(vR, cmap = 'gray')

#how well do our first 50 pc's do at approximating it?
comp = 0+Xbar
for i in range(50):
  vi = V[:,i]
  Ci = (vr-Xbar).dot(vi)
  comp = comp + Ci * vi

comp = np.reshape(comp, (28,28))
plt.imshow(comp, cmap = 'gray')

#"""For most images, I can get a rudimentary approximation with only 25 components. 
#To get a good, very clear approximation, 125 or 150 components seems sufficient. (Maybe even only 100 for some of the images.)"""

#top 16
idx = 1
for i in range(4):
  for j in range(4):
    vi = V[:,idx]
    vi = np.reshape(vi, (28,28))
    plt.subplot(4, 4, idx)
    plt.imshow(vi, cmap = 'gray')
    idx =idx + 1

#visualization
firstTwoCoeffs = np.zeros((60000, 2))
v1 = V[:,0]
v2 = V[:,1]
for i in range(60000):
  xi = X_train[i,:]
  firstTwoCoeffs[i,0] = (xi-Xbar).dot(v1)
  firstTwoCoeffs[i,1] = (xi-Xbar).dot(v2)

for idx in range(10):
  Yidx=np.argwhere(Y_train == 1)
  whr = np.argwhere(Yidx[:,1] == idx)
  plt.plot(firstTwoCoeffs[whr, 0], firstTwoCoeffs[whr,1],'.')
