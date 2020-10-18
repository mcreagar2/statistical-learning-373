#Author: Molly Creagar
#Project detailing different statistical learning classification algorithms
#uses data set Heart Disease UCI from Kaggle
#appends data into resultsDF (normalized data) or resultsDFraw (not normalized data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import timeit



def plotLine(a,xMin,xMax,yMin,yMax):
    #get the blue line between the clouds
    xVals = np.linspace(xMin,xMax,100)
    yVals = (-a[0] * xVals - a[2])/a[1]
    idxs =  np.where((yVals >= yMin) & (yVals <= yMax))
    plt.plot(xVals[idxs],yVals[idxs])
    
def sigmoid(u):
    
#    return np.exp(u)/(1.0 + np.exp(u))
    return 1.0/(1.0 + np.exp(-u))

def F(x, y, beta):
    #logL
    sumF = 0
    numRows = x.shape[0]
    for i in range(numRows):
        xi = x[i,:]
        term = sigmoid(np.vdot(xi,beta))
        val = y[i]*np.log(term + .001) + (1 - y[i])*np.log(1 - term + 0.001)
        #val = y[i]*np.log(np.exp((x[i,:].dot(beta))/(1+(x[i,:].dot(beta))))) + (1-y[i])*np.log(1-np.exp((x[i,:].dot(beta))/(1+(x[i,:].dot(beta)))))
        sumF = sumF + val

    return -sumF

def gradF(x, y, beta):
    #gradient of logL fn
    sumF = 0
    numRows, numCols = x.shape
    for i in range(numRows):
        xi = x[i,:]
        term = sigmoid(np.vdot(xi,beta))
        val = np.dot(xi.T, (y[i]-term))
        sumF = sumF + val        
    return -sumF


def logReg_GD(X, Y, t):
    maxIter = 5000
    numRows, numCols = X.shape
    beta_GD = np.zeros(numCols)
    cost = np.zeros(maxIter)
    xarr = np.zeros(maxIter)
    place = 1
    for i in range(maxIter):
        grad = gradF(X, Y, beta_GD)
        beta_GD = beta_GD - t*grad
        cost[i] = F(X, Y, beta_GD)
        xarr[i] = place
        place = place + 1
    fig2, ax2 = plt.subplots()
    ax2.semilogy(xarr, cost)
    ax2.set_title("Ordinary Gradient Descent Semilog-y Plot")
    print("Cost Returned from Ordinary Gradient Descent:", cost[maxIter-1])         
    return beta_GD

def logReg_StocGD(x, y, t, epoch):
    numRows, numCols = x.shape
    beta_SGD = np.zeros(numCols)
    costs = np.zeros(epoch)
    for ep in range(epoch):
    
        for i in np.random.permutation(numRows):
            xi = x[i,:]
            prod = np.vdot(xi, beta_SGD)
            beta_SGD = beta_SGD - t*((sigmoid(prod) - y[i])* xi)
            
        costs[ep] = F(x, y, beta_SGD)
 
    return (beta_SGD, costs)



##########
#reading in data
    
data = pd.read_csv('~/heart.csv')
list(data.columns.values)

X = data.copy()
y = X.iloc[:, -1].values
X = X.iloc[:, :-1]
numrows = X.shape[0]
X["Ones"] = np.ones(numrows)

Xmat = X.values.astype(float)
Xmat_notscaled = X.values.astype(float)

numExamples, numFeatures = Xmat.shape

for i in range(numFeatures-1):
    maxC = max(Xmat[:,i])
    minC = min(Xmat[:,i])
    Xmat[:,i]= (Xmat[:,i] - minC)/(maxC-minC)






resultsDF = pd.DataFrame(columns = ["Method", "Accuracy", "Time"] )
resultsDFraw = pd.DataFrame(columns = ["Method", "Accuracy", "Time"] )
#################
    ##Logistic Regression with Ordinary Gradient Descent
   
print("Logistic Regression with Ordinary Gradient Descent")
X_test, X_train, Y_test, Y_train = sk.model_selection.train_test_split(Xmat, y, test_size = 0.5, random_state = 41)
timeStart = timeit.default_timer()

betaLR = logReg_GD(X_train, Y_train, 0.001)
timeLR = timeit.default_timer() - timeStart

yhatLR = np.matmul(X_test, betaLR)

for i in range(len(yhatLR)):
    if yhatLR[i] > 0:
        yhatLR[i] = 1
    elif yhatLR[i] < 0:
        yhatLR[i] = 0

wrong = np.where(yhatLR != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongLR = k/tot
fracRightLR = 1 - fracWrongLR

print("Fraction Classified Incorrectly: ", fracWrongLR, \
      " Fraction Classified Correctly: ", fracRightLR )


LR = {'Method':["Ordinary Logistic Regression"], 'Accuracy':[fracRightLR], 'Time':[timeLR]}
LR = pd.DataFrame(LR)
resultsDF = resultsDF.append(LR)

#################
  ####Logistic Regression with Stochastic Gradient Descent

  
print("Logistic Regression with Stochastic Gradient Descent")
numEpochs = 80

timeStart = timeit.default_timer()
betaSt, costSGD = logReg_StocGD(X_train, Y_train, 0.01, numEpochs)
timeSGD = timeit.default_timer() - timeStart

fig, ax = plt.subplots()
ax.semilogy(costSGD)
ax.set_title("Stochastic Gradient Descent Semilog-y Plot")
    
yhatSt = np.matmul(X_test, betaSt)


for i in range(len(yhatSt)):
    if yhatSt[i] > 0:
        yhatSt[i] = 1
    elif yhatSt[i] < 0:
        yhatSt[i] = 0


wrong = np.where(yhatSt != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongSt = k/tot
fracRightSt = 1 - fracWrongSt

print("Fraction Classified Incorrectly: ", fracWrongSt, \
      " Fraction Classified Correctly: ", fracRightSt )

SGD = {'Method':["Stochastic Gradient Descent in Logistic Regression"], 'Accuracy':[fracRightSt], 'Time':[timeSGD]}
SGD = pd.DataFrame(SGD)
resultsDF = resultsDF.append(SGD)

#####
##K-Nearest Neighbors
print("K-Nearest Neighbors")
numNeighbors = 10


timeStart = timeit.default_timer()
knn = KNeighborsClassifier(n_neighbors = numNeighbors, metric = 'euclidean')
knn.fit(X_train, Y_train)

timeK = timeit.default_timer() - timeStart

Y_predKNN = knn.predict(X_test)
knn.score(X_test, Y_test)

confusion_matrix(Y_test, Y_predKNN)


wrong = np.where(Y_predKNN != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongK = k/tot
fracRightK = 1 - fracWrongK

print("Fraction Classified Incorrectly: ", fracWrongK, \
      " Fraction Classified Correctly: ", fracRightK)

KNN = {'Method':["K-Nearest Neighbors"], 'Accuracy':[fracRightK], 'Time':[timeK]}
KNN = pd.DataFrame(KNN)
resultsDF = resultsDF.append(KNN)

#####
#Decision Tree

print("Decision Tree")

timeStart = timeit.default_timer()
clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 41)
clf.fit(X_train, Y_train)
timeTREE = timeit.default_timer() - timeStart
Y_predTREE = clf.predict(X_test)

print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predTREE))


wrong = np.where(Y_predTREE != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongTR = k/tot
fracRightTR = 1 - fracWrongTR

print("Fraction Classified Incorrectly: ", fracWrongTR, \
      " Fraction Classified Correctly: ", fracRightTR)


TREE = {'Method':["Decision Tree"], 'Accuracy':[fracRightTR], 'Time':[timeTREE]}
TREE = pd.DataFrame(TREE)
resultsDF = resultsDF.append(TREE)

#######
#Random Forest


print("Random Forests")
timeStart = timeit.default_timer()
rf = RandomForestClassifier(n_estimators = 30, random_state=41)
rf.fit(X_train, Y_train)
timeRF = timeit.default_timer() - timeStart

Y_predRF = rf.predict(X_test)
print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predRF))

wrong = np.where(Y_predRF != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)

fracWrongRF = k/tot
fracRightRF = 1 - fracWrongRF

print("Fraction Classified Incorrectly: ", fracWrongRF, \
      " Fraction Classified Correctly: ", fracRightRF)

RF = {'Method':["Random Forests"], 'Accuracy':[fracRightRF], 'Time':[timeRF]}
RF = pd.DataFrame(RF)
resultsDF = resultsDF.append(RF)

####
#Support Vector Machine
print("SVM")

timeStart = timeit.default_timer()
#clf = sklearn.svm.SVC(kernel='linear')
clf = sklearn.svm.SVC(kernel='rbf',gamma=1.5)
#clf = sklearn.svm.SVC(kernel='poly',degree = 2, coef0 = 15)
clf.fit(X_train, Y_train)
timeSVM = timeit.default_timer() - timeStart


Y_predSVM = clf.predict(X_test)
print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predSVM))

wrong = np.where(Y_predSVM != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongSVM = k/tot
fracRightSVM = 1 - fracWrongSVM

print("Fraction Classified Incorrectly: ", fracWrongSVM, \
      " Fraction Classified Correctly: ", fracRightSVM)

SVM = {'Method':["Support Vector Machine"], 'Accuracy':[fracRightSVM], 'Time':[timeSVM]}
SVM = pd.DataFrame(SVM)
resultsDF = resultsDF.append(SVM)






#%%Now, we run the same analysis but without normalizing the data to start
#%%    


print("(Not Normalized) Logistic Regression with Ordinary Gradient Descent")
X_test, X_train, Y_test, Y_train = sk.model_selection.train_test_split(Xmat_notscaled, y, test_size = 0.5, random_state = 41)
timeStart = timeit.default_timer()

betaLR = logReg_GD(X_train, Y_train, 0.01)
timeLR = timeit.default_timer() - timeStart

yhatLR = np.matmul(X_test, betaLR)

for i in range(len(yhatLR)):
    if yhatLR[i] > 0:
        yhatLR[i] = 1
    elif yhatLR[i] < 0:
        yhatLR[i] = 0

wrong = np.where(yhatLR != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongLR = k/tot
fracRightLR = 1 - fracWrongLR

print("Fraction Classified Incorrectly: ", fracWrongLR, \
      " Fraction Classified Correctly: ", fracRightLR )


LR = {'Method':["Ordinary Logistic Regression"], 'Accuracy':[fracRightLR], 'Time':[timeLR]}
LR = pd.DataFrame(LR)
resultsDFraw = resultsDFraw.append(LR)

#################
  ####Logistic Regression with Stochastic Gradient Descent
#%%
  

print("(Not Normalized) Logistic Regression with Stochastic Gradient Descent")
numEpochs = 100

timeStart = timeit.default_timer()
betaSt, costSGD = logReg_StocGD(X_train, Y_train, 0.01, numEpochs)
timeSGD = timeit.default_timer() - timeStart

plt.figure()
plt.semilogy(costSGD)

yhatSt = np.matmul(X_test, betaSt)


for i in range(len(yhatSt)):
    if yhatSt[i] > 0:
        yhatSt[i] = 1
    elif yhatSt[i] < 0:
        yhatSt[i] = 0


wrong = np.where(yhatSt != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongSt = k/tot
fracRightSt = 1 - fracWrongSt

print("Fraction Classified Incorrectly: ", fracWrongSt, \
      " Fraction Classified Correctly: ", fracRightSt )

SGD = {'Method':["Stochastic Gradient Descent in Logistic Regression"], 'Accuracy':[fracRightSt], 'Time':[timeSGD]}
SGD = pd.DataFrame(SGD)
resultsDFraw = resultsDFraw.append(SGD)

#####
#%%

print("(Not Normalized) K-Nearest Neighbors (using numNeighbors=10)")
numNeighbors = 10


timeStart = timeit.default_timer()
knn = KNeighborsClassifier(n_neighbors = numNeighbors, metric = 'euclidean')
knn.fit(X_train, Y_train)

timeK = timeit.default_timer() - timeStart

Y_predKNN = knn.predict(X_test)
knn.score(X_test, Y_test)

confusion_matrix(Y_test, Y_predKNN)


wrong = np.where(Y_predKNN != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongK = k/tot
fracRightK = 1 - fracWrongK

print("Fraction Classified Incorrectly: ", fracWrongK, \
      " Fraction Classified Correctly: ", fracRightK)

KNN = {'Method':["K-Nearest Neighbors"], 'Accuracy':[fracRightK], 'Time':[timeK]}
KNN = pd.DataFrame(KNN)
resultsDFraw = resultsDFraw.append(KNN)

#####
#%%

print("(Not Normalized) Decision Tree")
timeStart = timeit.default_timer()
clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 41)
clf.fit(X_train, Y_train)
timeTREE = timeit.default_timer() - timeStart
Y_predTREE = clf.predict(X_test)

print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predTREE))


wrong = np.where(Y_predTREE != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongTR = k/tot
fracRightTR = 1 - fracWrongTR

print("Fraction Classified Incorrectly: ", fracWrongTR, \
      " Fraction Classified Correctly: ", fracRightTR)


TREE = {'Method':["Decision Tree"], 'Accuracy':[fracRightTR], 'Time':[timeTREE]}
TREE = pd.DataFrame(TREE)
resultsDFraw = resultsDFraw.append(TREE)

#######
#%%

print("(Not Normalized) Random Forest")
timeStart = timeit.default_timer()
rf = RandomForestClassifier(n_estimators = 500, random_state=41)
rf.fit(X_train, Y_train)
timeRF = timeit.default_timer() - timeStart

Y_predRF = rf.predict(X_test)
print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predRF))

wrong = np.where(Y_predRF != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongRF = k/tot
fracRightRF = 1 - fracWrongRF

print("Fraction Classified Incorrectly: ", fracWrongRF, \
      " Fraction Classified Correctly: ", fracRightRF)

RF = {'Method':["Random Forests"], 'Accuracy':[fracRightRF], 'Time':[timeRF]}
RF = pd.DataFrame(RF)
resultsDFraw = resultsDFraw.append(RF)

####
#%%

print("(Not Normalized) SVM")
timeStart = timeit.default_timer()
clf = sklearn.svm.SVC(kernel='linear')
#clf = sklearn.svm.SVC(kernel='rbf',gamma=1)
#clf = sklearn.svm.SVC(kernel='poly',degree = 2, coef0 = 5)
clf.fit(X_train, Y_train)
timeSVM = timeit.default_timer() - timeStart


Y_predSVM = clf.predict(X_test)
print("Accuracy:", sk.metrics.accuracy_score(Y_test,Y_predSVM))

wrong = np.where(Y_predSVM != Y_test)[0]
k = float(len(wrong))
tot = len(Y_test)
fracWrongSVM = k/tot
fracRightSVM = 1 - fracWrongSVM

print("Fraction Classified Incorrectly: ", fracWrongSVM, \
      " Fraction Classified Correctly: ", fracRightSVM)

SVM = {'Method':["Support Vector Machine"], 'Accuracy':[fracRightSVM], 'Time':[timeSVM]}
SVM = pd.DataFrame(SVM)
resultsDFraw = resultsDFraw.append(SVM)
