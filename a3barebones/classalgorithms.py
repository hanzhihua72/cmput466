import numpy as np

import MLCourse.utilities as utils

# Susy: ~50 error

# Return the Hamming distance between string1 and string2.
# string1 and string2 should be the same length.
def hamming(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance


class Classifier:
    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error


class LinearRegressionClass(Classifier):
    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + \
            self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error


class NaiveBayes(Classifier):
    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items(
            {'usecolumnones': False}, parameters)

    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
            numsamples = Xtrain.shape[0]

            if self.params['usecolumnones']:
                Xtrain = np.concatenate([np.ones((numsamples, 1)), Xtrain], axis=1)

            numfeatures = Xtrain.shape[1]

            # print(Xtrain.shape, ytrain.shape)

            # Compute the prior
            self.prior = np.zeros((2))
            for c in range(self.numclasses):
                self.prior[c] = len(
                    [i for i in range(ytrain.shape[0]) if ytrain[i, 0] == c])/numsamples

            self.means = np.zeros((numfeatures, self.numclasses))
            self.var = np.zeros((numfeatures, self.numclasses))

            for c in range(self.numclasses):
                for j in range(numfeatures):
                    # indicies where c = 1
                    index = [i for i in range(
                        ytrain.shape[0]) if ytrain[i, 0] == c]
                    # print(Xtrain[index, j].shape)
                    self.means[j, c] = np.mean(Xtrain[index, j])
                    self.var[j, c] = np.var(Xtrain[index, j])

        else:
            raise Exception('Can only handle binary classification')

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]

        if self.params['usecolumnones']:
            Xtest = np.concatenate([np.ones((numsamples, 1)), Xtest], axis=1)

        numfeatures = Xtest.shape[1]
        predictions = []

        for i in range(numsamples):
            ypred = self.prior.copy()

            for c in range(self.numclasses):
                for j in range(numfeatures):
                    ypred[c] *= utils.gaussian_pdf(Xtest[i, j],
                                                   self.means[j, c], np.sqrt(self.var[j, c]))

            predictions.append(np.argmax(ypred))
        return np.reshape(predictions, [numsamples, 1])

# Susy: ~23 error


class LogisticReg(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items(
            {'stepsize': 0.01, 'epochs': 100}, parameters)

    def learn(self, X, y):
        self.weights = np.random.randn(X.shape[1], 1)

        maxiter = self.params["epochs"]
        for i in range(1, maxiter):
            eta = self.params["stepsize"]/i
            self.weights -= eta*X.T @ (utils.sigmoid(X @ self.weights) - y)

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        numfeatures = Xtest.shape[1]
        predictions = []

        for i in range(numsamples):
            prob = utils.sigmoid(self.weights.T @ Xtest[i, :])
            if prob < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)

        return np.reshape(predictions, [numsamples, 1])


# Susy: ~23 error (4 hidden units)

class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception(
                'NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

    def learn(self, Xtrain, ytrain):
        numfeatures = Xtrain.shape[1]
        numsamples = Xtrain.shape[0]

        self.w2 = np.random.randn(numfeatures, self.params['nh'])
        self.w1 = np.random.randn(self.params['nh'], 1)

        maxiter = self.params["epochs"]

        delta_1 = np.zeros((self.params['nh']))
        delta_2 = np.zeros((self.params['nh']))

        x_shuffle = np.concatenate([Xtrain, ytrain], axis=1)
        for i in range(maxiter):

            np.random.shuffle(x_shuffle)
            Xtrain = x_shuffle[:, :numfeatures]
            ytrain = x_shuffle[:, numfeatures]

            for n in range(numsamples):
                x = Xtrain[n, :]

                h = self.transfer(x @ self.w2)
                yhat = self.transfer(h @ self.w1)

                for k in range(self.params['nh']):
                    delta_1 = yhat - ytrain[n]
                    self.w1[k, 0] -= self.params['stepsize']*delta_1*h[k]

                for k in range(self.params['nh']):
                    for j in range(numfeatures):
                        delta_2[k] = (self.w1[k, 0] * delta_1)*h[k]*(1-h[k])
                        self.w2[j, k] -= self.params['stepsize']*delta_2[k]*x[j]
            print('NN epoch training:'+str(round(i/self.params['epochs']*100, 1))+'%')

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        numfeatures = Xtest.shape[1]
        predictions = []

        for i in range(numsamples):
            x = Xtest[i]

            h = self.transfer(x @ self.w2)
            yhat = self.transfer(h @ self.w1)

            if yhat < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.reshape(predictions, [numsamples, 1])

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi, inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo, ah)).T

        return (
            ah,  # shape: [nh, samples]
            ao,  # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        pass

# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)


class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 1000,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        #print(X)
        numfeatures = X.shape[1]
        numsamples = X.shape[0]

        K = np.zeros((numsamples, self.params['centers']))
        self.weights = np.random.randn(self.params['centers'], 1)
        index = np.random.choice(numsamples, size=self.params['centers'])
        self.centers = X[index].copy()
        
        for n in range(numsamples):
            for i, C in enumerate(self.centers):
                K[n, i] = X[n] @ C

        assert ((K @ self.weights).shape == y.shape)

        maxiter = self.params["epochs"]
        for i in range(1, maxiter):
            eta = self.params["stepsize"]/i
            self.weights -= eta * \
                K.T @ (utils.sigmoid(K @ self.weights) - y)
        

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        numfeatures = Xtest.shape[1]
        predictions = []

        K = np.zeros((numsamples, self.params['centers']))

        for n in range(numsamples):
            for i, C in enumerate(self.centers):
                K[n, i] = Xtest[n] @ C

        for n in range(numsamples):
            prob = utils.sigmoid(K[n] @ self.weights)
            if prob < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)

        return np.reshape(predictions, [numsamples, 1])



# New function to solely deal with Census
class KernelLogisticRegressionCensus(LogisticReg):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 1000,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        numfeatures = len(X[0])
        numsamples = X.shape[0]

        K = np.zeros((numsamples, self.params['centers']))
        self.weights = np.random.randn(self.params['centers'], 1)
        index = np.random.choice(numsamples, size=self.params['centers'])
        self.centers = X[index].copy()
        
        for n in range(numsamples):
            for i, C in enumerate(self.centers):
                K[n, i] = hamming(X[n], C)

        assert ((K @ self.weights).shape == y.shape)

        maxiter = self.params["epochs"]
        for i in range(1, maxiter):
            eta = self.params["stepsize"]/i
            self.weights -= eta * \
                K.T @ (utils.sigmoid(K @ self.weights) - y)
        

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        numfeatures = len(Xtest[0])
        predictions = []

        K = np.zeros((numsamples, self.params['centers']))

        for n in range(numsamples):
            for i, C in enumerate(self.centers):
                K[n, i] = hamming(Xtest[n] , C)

        for n in range(numsamples):
            prob = utils.sigmoid(K[n] @ self.weights)
            if prob < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)

        return np.reshape(predictions, [numsamples, 1])