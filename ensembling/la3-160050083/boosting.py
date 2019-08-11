import util
import numpy as np
import sys
import random
import math

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """

        n = len(trainingData)

        weights = [1.0/n]*n

        for i in range(self.boosting_iterations):
            # print "iteration ",i
            

            self.classifiers[i].train(trainingData,trainingLabels,weights)
            pred = self.classifiers[i].classify(trainingData)

            error = 0.0
            for j in range(n):
                if pred[j]!=trainingLabels[j]:
                    error += weights[j]

            error = error/(1.0-error)

            for j in range(n):
                if pred[j]==trainingLabels[j]:
                    weights[j] *= error

            weights = util.normalize(weights)
            self.alphas[i] = - math.log10(error)




    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """
        poll = np.zeros( (self.boosting_iterations,len(data)) )

        for i in range(self.boosting_iterations):
            poll[i]=np.array(self.classifiers[i].classify(data))*self.alphas[i]

        r = np.sign(np.sum(poll,axis=0))
        return r