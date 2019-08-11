import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *

# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	in_nodes = len(XTrain[0])
	alpha = 0.5
	batchSize = 20
	epochs = 30
	out_nodes = len(YTrain[0])
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(in_nodes,5))
	nn1.addLayer(FullyConnectedLayer(5,out_nodes))
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	# print(pred,YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	in_nodes = len(XTrain[0])
	alpha = 0.1
	batchSize = 10
	epochs = 30
	out_nodes = len(YTrain[0])
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(in_nodes,2))
	nn1.addLayer(FullyConnectedLayer(2,out_nodes))

	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	in_nodes = len(XTrain[0])
	alpha = 0.05
	batchSize = 8
	epochs = 30
	out_nodes = len(YTrain[0])
	H1 = 12
	H2 = 12
	# print('mnist','H1',H1,'H2',H2)
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(in_nodes,H1))
	nn1.addLayer(FullyConnectedLayer(H1,H2))
	nn1.addLayer(FullyConnectedLayer(H2,out_nodes))
	# raise NotImplementedError	
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	t = 1000     # max 1000
	p = 100      # max 100
	XTrain = XTrain[0:5*t,:,:,:]
	XVal = XVal[0:10*p,:,:,:]
	XTest = XTest[0:10*p,:,:,:]
	YVal = YVal[0:10*p,:]
	YTest = YTest[0:10*p:]
	# YTrain = YTrain[0:5*t,:]
	
	modelName = 'model_full.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	in_nodes = XTrain[0].shape
	# print(in_nodes)
	alpha = 0.1
	batchSize = 20
	epochs = 30
	out_nodes = len(YTrain[0])

	nn1 = nn.NeuralNetwork(10, alpha, batchSize, epochs)
	nn1.addLayer(ConvolutionLayer([3,32,32], [4,4], 16, 2))
	nn1.addLayer(AvgPoolingLayer([16,15,15], [3,3], 3))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(400,100))
	nn1.addLayer(FullyConnectedLayer(100,10))

	# raise NotImplementedError	
	###################################################
	# return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION


	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)