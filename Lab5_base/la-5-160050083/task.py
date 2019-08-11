import numpy as np
from utils import *
import time

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	n = X.shape[1]
	m = X.shape[0]
	T = np.zeros((m,1))
	T[:,0]=1;
	for i in range(1,n):
		if(type(X[1,i])==type('string')):
			Z=one_hot_encode(X[:,i],np.unique(X[:,i]))
			T = np.append(T,Z,axis=1)
		else :
			Z = X[:,i:i+1]
			Z = (Z - Z.mean())/Z.std()
			T = np.append(T,Z,axis=1)
	return T.astype('float64'),Y.astype('float64');

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	grad = 2*(np.matmul(X,W)-Y)+2*_lambda*W;
	return grad

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	dims = X.shape[1]
	n = X.shape[0]
	W = np.zeros((dims,1))
	XTX = np.matmul(np.transpose(X),X)
	XTY = np.matmul(np.transpose(X),Y)
	for _ in range(max_iter):
		grad = grad_ridge(W,XTX,XTY,_lambda)
		if(np.linalg.norm(grad)<epsilon):
			return W
		W = W - lr*grad
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	l=[]
	m = X.shape[0]
	dims = X.shape[1]
	# print(X.shape)
	partition_size = m//k
	sz=len(lambdas)
	# print(lambdas)
	for j in range(sz):
		t = np.zeros((k,1))
		# print(j,"iter")
		for i in range(k):
			idx = np.zeros(m)
			idx[i*partition_size:(i+1)*partition_size] = 1
			train_data = X[idx[:]==0,:]
			train_Y = Y[idx[:]==0]
			test_data = X[idx==1]
			test_Y = Y[idx[:]==1]
			# print(train_data.shape,train_Y.shape)
			# print(lambdas[j])
			W = algo(train_data,train_Y,lambdas[j])
			t[i]=(sse(test_data,test_Y,W))
			# print(t[i]/partition_size)
		l.append(t.mean())
	return l

	pass

def coord_grad_descent(X, Y, _lambda, max_iter=200):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	# t=time.time()
	dims = X.shape[1]
	n = X.shape[0]
	W = np.zeros((dims,1))
	for j in range(max_iter):
		for i in range(dims):
			A = X[:,i:i+1]
			B = np.matmul(X,W)-Y-W[i]*A
			a = 2*(np.sum(A*A))
			if(a!=0):
				b = 2*(np.sum(B*A))
				w1 = (-b-_lambda)/a
				w2 = (-b+_lambda)/a
				if w1>=0:
					W[i] = w1
				elif w2 <=0:
					W[i] = w2
				else :
					W[i] = 0
	# print("elapsed time:",time.time()-t)
	return W
	pass

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	# tried values of lambda 
	# lambdas = [0,1e-10,1e-5,1e-2,1e-1,1,10,100]#1,0.1,0.01,0.0001,] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# lambdas = [1e-1,0.5,1,2,5,10,20,100]#1,0.1,0.01,0.0001,] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# lambdas = [5,7.5,10,12.5,15,20,22]
	# lambdas = [11,11.5,12,12.5,13,13.5]
	# lambdas = [12.2,12.4,12.5,12.6,12.8]
	# lambdas = [12.2,12.3,12.4,12.5,12.6]
	#[170318027348.8386, 170317473054.13617, 170317231412.72385, 170317295154.2621, 170317657211.6348]	
	lambdas = [4,7,12.4,18,24]
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	for i in range(len(lambdas)):
		print(lambdas[i],"\t",scores[i])
	plot_kfold(lambdas, scores)
	lambda_opt_ridge = 12.4
	weigths_ridge = ridge_grad_descent(trainX,trainY,lambda_opt_ridge)
	print("Ridge weights",weigths_ridge)
	error = sse(testX,testY,weigths_ridge)
	print("sse using ridge-regression",error)
	
	# tried values of lambda 
	# lambdas = [0,1e-10,1e-5,1e-2,1e-1,1,10,100]#1,0.1,0.01,0.0001,] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# lambdas = [100,1000,1e4,1e6] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# lambdas = [1e5,1e6,0.5*1e7,1e7]
	# lambdas = [0.5*1e6,0.8*1e6,1e6,3*1e6,5*1e6]
	# lambdas = [0.3*1e6,0.4*1e6,0.5*1e6,0.55*1e6,0.6*1e6,0.8*1e6]
	# lambdas = [0.32*1e6,0.35*1e6,0.4*1e6,0.43*1e6,0.47*1e6]
	# lambdas = [0.41*1e6,0.42*1e6,0.43*1e6]#0.44*1e6,0.45*1e6,0.46*1e6,0.47*1e6,0.48*1e6,0.49*1e6]
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	# 0.42*1e6 168789684760.05093
	lambdas = [0.42*1e4,0.42*1e5,0.42*1e6,0.6*1e6,0.8*1e6]
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	for i in range(len(lambdas)):
		print(lambdas[i],"\t",scores[i])
	plot_kfold(lambdas, scores)

	lambda_opt_ridge = 0.42*1e6
	weigths_lasso = coord_grad_descent(trainX,trainY,lambda_opt_ridge)
	error = sse(testX,testY,weigths_lasso)
	print("sse using lasso-regression",error)
	print("Ridge weights",weigths_lasso)

