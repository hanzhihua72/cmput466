import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs

import pprint


def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100


def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))


""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""


def cross_validate(K, X, Y, Algorithm, parameters):

    all_errors = np.zeros((len(parameters), K))

    for k in range(K):
        Xarrays = np.split(X, K)
        Yarrays = np.split(Y, K)

        Xtest = Xarrays.pop(k)
        ytest = Yarrays.pop(k)

        Xtrain = np.concatenate(Xarrays)
        ytrain = np.concatenate(Yarrays)

        for i, params in enumerate(parameters):
            algorithm = Algorithm(params)
            algorithm.learn(Xtrain, ytrain)
            predictions = algorithm.predict(Xtest)

            all_errors[i, k] = geterror(ytest, predictions)
            print(f'{Algorithm.__name__} Cross validate parameters : {params} error: {all_errors[i, k]} run {k+1}/{K}, parameters {i+1}/{len(parameters)}')
    
    avg_errors = np.mean(all_errors, axis=1)
    avg_std = np.std(all_errors, axis=1)/np.sqrt(K)

    runs = []
    lowest_error = np.inf
    best_parameters = parameters[0]
    for i, params in enumerate(parameters):
        runs.append({'name': Algorithm.__name__, 'params': params,
                     'average_error': avg_errors[i],
                     'standard_error': avg_std[i],
                     })
        if avg_errors[i] < lowest_error:
            lowest_error = avg_errors[i]
            best_parameters = parameters[i]

    pprint.pprint(runs)

    return best_parameters


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=1,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset

    classalgs = {
        # 'Random': algs.Classifier,
         'Naive Bayes': algs.NaiveBayes,
        # 'Linear Regression': algs.LinearRegressionClass,
         'Logistic Regression': algs.LogisticReg,
         'Neural Network': algs.NeuralNet,
         'Kernel Logistic Regression': algs.KernelLogisticRegression,
        # 'Kernel Logistic Regression Census': algs.KernelLogisticRegressionCensus,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            {'usecolumnones': True},
            # second set of parameters to try
            {'usecolumnones': False},
        ],
        'Logistic Regression': [
            {'stepsize': 1},
            {'stepsize': 0.1},
            {'stepsize': 0.01},
        ],
        'Neural Network': [
            {'epochs': 100, 'nh': 4},  # MUST BE RUN ONE AT A TIME
            {'epochs': 100, 'nh': 8},
            {'epochs': 100, 'nh': 16},
            #{'epochs': 100, 'nh': 32},
        ],
        'Kernel Logistic Regression': [
            {'centers': 10, 'stepsize': 0.01},
            {'centers': 20, 'stepsize': 0.01},
            {'centers': 40, 'stepsize': 0.01},
            {'centers': 80, 'stepsize': 0.01},
        ],
        'Kernel Logistic Regression Census': [
            {'centers': 10, 'stepsize': 0.01},
            {'centers': 20, 'stepsize': 0.01},
            {'centers': 40, 'stepsize': 0.01},
            {'centers': 80, 'stepsize': 0.01},
        ],
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize, testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [None])
            best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)

        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            print(f'Best parameters for {learnername} :')
            pprint.pprint(params)
            learner = Learner(params)
            learner.learn(Xtrain, Ytrain)
            errors[learnername] = geterror(Ytest, learner.predict(Xtest))

    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername])/np.sqrt(numruns)
        print('Average error for ' + learnername + ': ' + str(aveerror) + '+-' + str(stderror))
