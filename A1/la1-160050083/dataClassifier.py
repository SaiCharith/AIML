# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This file contains feature extraction methods and harness
# code for data classification

import perceptron1vr
import perceptron1v1
import samples
import sys
import util
import numpy
import math


sys.setrecursionlimit(3000)

TRAIN_SET_SIZE = 80000
TEST_SET_SIZE = 20000 
DATUM_WIDTH=50
DATUM_HEIGHT=50

def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    features = util.Counter()
    for i in range(len(datum)):
        features[i] = datum[i]
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (a unit of data)

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    
    def getperimiter(D):
        p=0
        for i in range(1,DATUM_HEIGHT-1):
            for j in range(1,DATUM_HEIGHT-1):
                if D[i][j]==1:
                    l = D[i+1][j]+D[i][j+1]+D[i-1][j]+D[i][j-1]
                    if l<4:
                        p+=1
        return p

    def getPixelVal(x, y):
        """
        Helper Function to return the pixel value at location x, y
        1 : black
        0 : white
        Refer to the basicFeatureExtractorDigit function for more Details
        """
        return datum[x * DATUM_HEIGHT + y]

    #getting bounding box min x,y max x,y which are 1
    def getbbox(D):
        l=[[DATUM_HEIGHT,DATUM_WIDTH],[0,0]]
        for i in range(DATUM_HEIGHT):
            for j in range(DATUM_HEIGHT):
                if D[i][j]==1:
                    l[0][0] = min(i,l[0][0])
                    l[0][1] = min(j,l[0][1])
                    l[1][0] = max(i,l[0][0])
                    l[1][1] = max(j,l[1][1])
        return (l[1][1]-l[0][1])*(l[1][0]-l[0][0])


    def getboundpointstop(D):
        for i in range(0,DATUM_HEIGHT):
            for j in range(0,DATUM_WIDTH):
                if D[i][j]==1:
                    return [i,j] 
    def getboundpointsleft(D):
        for j in range(0,DATUM_WIDTH):
            for i in range(0,DATUM_HEIGHT):
                if D[i][j]==1:
                    return [i,j]
    def getboundpointsbottom(D):
        for i in range(0,DATUM_HEIGHT):
            for j in range(0,DATUM_WIDTH):
                if D[DATUM_HEIGHT-i-1][j]==1:
                    return [DATUM_HEIGHT-i-1,j]  
    def getboundpointsright(D):
        for j in range(0,DATUM_WIDTH):
            for i in range(0,DATUM_HEIGHT):
                if D[i][DATUM_WIDTH-j-1]==1:
                    return [i,DATUM_WIDTH-j-1] 

    def dist(x1,x2,x3,x4):
        return math.sqrt((x1-x3)*(x1-x3)+(x2-x4)*(x2-x4))

    def getarea(x1,y1,x2,y2,x3,y3):
        a = 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
        return abs(a)

    features = util.Counter()

    "*** YOUR CODE HERE ***"
    D = numpy.reshape(datum,(DATUM_HEIGHT,DATUM_WIDTH))
    areabbox = getbbox(D) # area enclosed by bounding box
    area = D.sum() #area of the figure

    # bounding points on all 4 directions
    [x1,y1] = getboundpointstop(D)
    [x2,y2] = getboundpointsleft(D)
    [x3,y3] = getboundpointsbottom(D)
    [x4,y4] = getboundpointsright(D)

    #area enclosed by bounding points
    area_bb = getarea(x1,y1,x2,y2,x3,y3)+getarea(x1,y1,x3,y3,x4,y4)
    #Perimeter of figure
    p = getperimiter(D)

    # minlength of boinding points side
    m=min(min(dist(x1,y1,x2,y2),dist(x2,y2,x3,y3)),min(dist(x3,y3,x4,y4),dist(x4,y4,x1,y1)))
    # maxlength of bounding points side
    l= max(max(dist(x1,y1,x2,y2),dist(x2,y2,x3,y3)),max(dist(x3,y3,x4,y4),max(x4,y4,x1,y1)))

    #bias
    #reletive area enclosed by figure wrt bbox
    features[0] = (1.0*area)/areabbox
    #reletive area enclosed by bounding points wrt figure
    features[1] = (1.0*area_bb)/areabbox
    features[2] = 1.0*m/p
    features[3] = 1.0*l/p
    # if m==0:
    #     print x1,y1,x2,y2,x3,y3,x4,y4

    return features

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c 1vr -t 1000 -f -s 1000
                  - would run the perceptron1vr classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the digits dataset, would test the classifier on the test data of 1000 examples
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['1vr', '1v1'], default='1vr')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAIN_SET_SIZE, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-v', '--validate', help=default("Whether to validate when training (for graphs)"), default=False, action="store_true")
    parser.add_option('-d', '--dataset', help=default("Specifies the data set to use"), choices=['d1', 'd2'], default='d1')
    parser.add_option('-k', '--classes', help=default("Specifies the number of classes"), default=10, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "classifier:\t\t" + options.classifier
    print "using enhanced features?:\t" + str(options.features)
    print "training set size:\t" + str(options.training)

    if (options.features):
        featureFunction = enhancedFeatureExtractorDigit
    else:
        featureFunction = basicFeatureExtractorDigit
    
    legalLabels = range(options.classes)

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if(options.classifier == "1vr"):
       classifier = perceptron1vr.Perceptron1vrClassifier(legalLabels,options.iterations)
    elif(options.classifier == "1v1"):
        classifier = perceptron1v1.Perceptron1v1Classifier(legalLabels, options.iterations)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING
        sys.exit(2)


    args['classifier'] = classifier
    args['featureFunction'] = featureFunction

    return args, options

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']

    # Load data
    dataset = options.dataset
    numTraining = options.training
    numTest = options.test

    if dataset == 'd1':
        rawTrainingData = samples.loadDataFile("data/D1/training_data", numTraining)
        trainingLabels = samples.loadLabelsFile("data/D1/training_labels", numTraining)
        rawTestData = samples.loadDataFile("data/D1/test_data", numTest)
        testLabels = samples.loadLabelsFile("data/D1/test_labels", numTest)

    else:
        rawTrainingData = samples.loadDataFile("data/D2/training_data", numTraining)
        trainingLabels = samples.loadLabelsFile("data/D2/training_labels", numTraining)
        rawTestData = samples.loadDataFile("data/D2/test_data", numTest)
        testLabels = samples.loadLabelsFile("data/D2/test_labels", numTest)

    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, testData, testLabels, options.validate)

    guesses = classifier.classify(trainingData)
    correct = [guesses[i] == trainingLabels[i] for i in range(len(trainingLabels))].count(True)
    
    if(options.classifier == "1vr"):
        f = open("perceptron1vr_train.csv","a")
        f.write(str(len(trainingData))+","+str(100*correct/(1.0*(len(trainingData))))+'\n')
        f.close()
    
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    
    if(options.classifier == "1vr"):
        f = open("perceptron1vr_test.csv","a")
        f.write(str(len(trainingData))+","+str(100*correct/(1.0*(len(testData))))+'\n')
        f.close()
        
if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)