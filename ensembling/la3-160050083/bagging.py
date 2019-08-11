import util
import numpy as np
import sys
import random
import math
PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers

        self.legalLabels = legalLabels
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]
        # print(legalLabels)

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        n = len(trainingData)
        fulldata = zip(trainingData,trainingLabels)

        # print len(fulldata[0])
        distribution = list([1])*n
        m = int(n*self.ratio)

        for i in range(self.num_classifiers):
            samples = util.nSample(distribution,fulldata,m)
            RandtarinData = [x for (x,v) in samples]
            Randlabels = [y for (x,y) in samples] 
            
            self.classifiers[i].train(RandtarinData,Randlabels)


    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        poll = np.zeros( (self.num_classifiers,len(data)) )

        for i in range(self.num_classifiers):
            poll[i]=np.array(self.classifiers[i].classify(data))

        r = np.sign(np.sum(poll,axis=0)+random.random()/2.0)
        return r




