import numpy as np
import math
import time

import MLCourse.utilities as utils

# -------------
# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0,
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------

class RidgeLinearRegression(Regressor):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5,
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest
    
class LassoRegression(Regressor):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5,
            'features': [1,2,3,4,5],
            'tol': 10e-4
        }, parameters)

    def learn(self, Xtrain, ytrain):
        n = Xtrain.shape[0]
        d = Xtrain.shape[1]
        self.weights = np.random.rand(Xtrain.shape[1])

        error = np.infty

        XX = (1/n)*Xtrain.T @ Xtrain
        Xy = (1/n)*Xtrain.T @ ytrain

        self.stepsize = 1/(2*np.linalg.norm(XX,ord="fro"))

        while np.abs(self.cost_weights(Xtrain, ytrain) - error) > self.params['tol']:
            
            #print(np.abs(self.cost_weights(Xtrain, ytrain) - error))
            error = self.cost_weights(Xtrain, ytrain)

            prox_argument = self.weights - self.stepsize*XX @ self.weights + self.stepsize*Xy
            for i in range(len(self.weights)):
                self.weights[i] = self.prox(prox_argument[i])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

    def prox(self, wi):
        if wi > self.stepsize*self.params['regwgt']:
            return wi - self.stepsize*self.params['regwgt']
        elif np.abs(wi) < self.stepsize*self.params['regwgt']:
            return 0
        elif wi < -self.stepsize*self.params['regwgt']:
            return wi + self.stepsize*self.params['regwgt']
    
    def cost_weights(self, Xtrain, ytrain):
        return np.linalg.norm(Xtrain @ self.weights - ytrain)**2 + self.params['regwgt']*np.linalg.norm(self.weights, ord=1)
    
class StochasticLinearRegression(Regressor):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'epochs': 1000,
            'stepsize': 0.01
        }, parameters)

    def learn(self, Xtrain, ytrain):
        n = Xtrain.shape[0]
        d = Xtrain.shape[1]
        self.weights = np.random.rand(Xtrain.shape[1])
        
        x_shuffle = np.c_[Xtrain, ytrain]
        for i in range(1, self.params['epochs']):

            np.random.shuffle(x_shuffle)
            Xtrain = x_shuffle[:, 0:385]
            ytrain = x_shuffle[:, 385]
            
            #print(i)
            for j in range(n):
                g = (Xtrain[j, :].T @ self.weights - ytrain[j])*Xtrain[j, :]
                self.weights = self.weights - (self.params['stepsize']/i)*g

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class BatchLinearRegression(Regressor):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'tol': 10e-4,
            'stepsize': 0.01,
            'maxiter': 1e4,
        }, parameters)

    def learn(self, Xtrain, ytrain):
        self.stepsize = self.params['stepsize']
        n = Xtrain.shape[0]
        d = Xtrain.shape[1]
        w = np.random.rand(Xtrain.shape[1])
        
        self.cost = []
        self.time = []
        start = time.time()
        
        self.process = 0
        
        while np.abs(self.cost_weights(w, Xtrain, ytrain)) > self.params['tol']:
            error = self.cost_weights(w, Xtrain, ytrain)
            
            self.cost.append(error)
            self.time.append(time.time()-start)
            
            g = self.grad_cost_weights(w, Xtrain, ytrain)
            self.stepsize = self.line_search(w, g, Xtrain, ytrain)
            w = w - self.stepsize*g
            
            self.process += 1
            
            #print(self.cost_weights(w, Xtrain, ytrain), self.stepsize)
            if self.stepsize == 0:
                break 
        self.weights = w

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
    def cost_weights(self,w, Xtrain, ytrain):
        return (1/(2*Xtrain.shape[0]))*np.linalg.norm(Xtrain @ w - ytrain)**2
    
    def grad_cost_weights(self,w, Xtrain, ytrain):
        return (1/(Xtrain.shape[0]))*Xtrain.T @ (Xtrain @ w - ytrain)
    
    def line_search(self, wt, g, Xtrain, ytrain):
        tau = 0.7
        tol = 10e-4
        self.iter = 0
        w = wt
        obj = self.cost_weights(w, Xtrain, ytrain)

        while self.iter < self.params['maxiter']:
            w = wt - self.stepsize*g
            if self.cost_weights(w, Xtrain, ytrain) < np.abs(obj - tol):
                break
            else:
                self.stepsize = tau*self.stepsize 
                self.iter += 1
                #print(self.iter)
        if self.iter == self.params['maxiter']:
            wt = 0
            self.stepsize = 0
            return self.stepsize
        return self.stepsize