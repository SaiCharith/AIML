
import numpy as np
from copy import deepcopy

class Transition:
	def __init__(self, next_state, reward, probability):
		self.next_state = next_state
		self.reward = reward
		self.probability = probability



def getTransitions(mdpPath):
	data = []
	with open(mdpPath) as inputfile:
	    for line in inputfile:
	        data.append(line.strip().split(' '))
	# print(data)
	numstates = int(data[0][1])
	numactions = int(data[1][1])
	startstate = int(data[2][1])
	endstateslist = [int(m) for m in data[3][1:]]
	endstates = np.zeros((numstates,1),dtype='int')
	TransitionList = [[[] for j in range(numactions)] for i in range(numstates)]
	for i in endstateslist:
		if(i>=0):
			endstates[i] = 1
	for i in range(4,len(data)-1):
		if endstates[int(data[i][1])]!=1:
			if float(data[i][5]) !=0:
				TransitionList[int(data[i][1])][int(data[i][2])].append(Transition(int(data[i][3]),float(data[i][4]),float(data[i][5])))
	gamma = float(data[-1][2])

	return numstates,numactions,startstate,endstates,TransitionList,gamma

def has_Converged(V,V1,epsilon=1e-17):
	print(V,np.absolute(V-V1).max())
	return np.absolute(V-V1).max()<=epsilon

def findOptimal1(numstates,numactions,TransitionList,gamma):

	V = np.zeros(numstates)
	p = np.zeros(numstates,dtype='int')
	iters = 0
	epsilon =1e-16

	while True:
		has_Converged = True
		V1 = deepcopy(V)
		for i in range(numstates):
			sz = len(TransitionList[i])
			t = float('-inf')
			for j in range(sz):
				s = 0.0
				for k in range(len(TransitionList[i][j])):
					s = s + TransitionList[i][j][k].probability*(TransitionList[i][j][k].reward+gamma*V[TransitionList[i][j][k].next_state])
				if(t<=s and len(TransitionList[i][j])>0):
					t=s
			if(t!=float('-inf')):
				if(abs(V[i]-t)>=epsilon):
					has_Converged = False
					# if has_Converged == False:
					# 	print(abs(V[i]-t))
					V1[i] = t
		iters = iters+1
		if has_Converged :
			break
		V = V1

	for i in range(numstates):
		sz = len(TransitionList[i])
		t = float('-inf')
		tj = None
		for j in range(sz):
			s = 0.0
			for k in range(len(TransitionList[i][j])):
				s = s + TransitionList[i][j][k].probability*(TransitionList[i][j][k].reward+gamma*V[TransitionList[i][j][k].next_state])
			if(t<=s and len(TransitionList[i][j])>0):
				t=s
				tj=j
		if(t!=float('-inf')):
			p[i] = tj

	return V,p,iters

