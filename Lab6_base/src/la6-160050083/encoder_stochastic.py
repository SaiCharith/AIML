import argparse
import time
from mdphelp import *

class Transition:
	def __init__(self, state, action, next_state, reward, probability):
		self.state = state
		self.action = action
		self.next_state = next_state
		self.reward = reward
		self.probability = probability

def createMdp(mazePath,p):
	data = []
	with open(mazePath) as inputfile:
	    for line in inputfile:
	        data.append(line.strip().split(' '))

	Data = [['1' for _ in range(len(data[0])+2)] for _ in range(len(data)+2)]
	for i in range(len(data)):
		for j in range(len(data[0])):
			Data[i+1][j+1] = data[i][j] 
	data = Data
	stateNum = 0
	endstate = -1
	startstate = -1
	TransitionList = []
	for i in range(1,len(data)-1):
		for j in range(1,len(data[i])-1):
			if(data[i][j]!='1'):
				if(data[i][j]=='3'):
					endstate = stateNum
				if(data[i][j]=='2'):
					startstate = stateNum
				data[i][j] = stateNum
				stateNum = stateNum + 1
				
	stateNum = 0

	def fillTransition(i,j,l,dir,act,stateNum):
		# print(dir,l)
		for elem in l:
			if dir==elem:
				if(data[i+elem[0]][j+elem[1]]==endstate):
					TransitionList.append(Transition(stateNum,act,endstate,0,p+(1-p)/len(l)))
				else:
					TransitionList.append(Transition(stateNum,act,data[i+elem[0]][j+elem[1]],-1,p+(1-p)/len(l)))
			elif(p!=1.0):
				if(data[i+elem[0]][j+elem[1]]==endstate):
					TransitionList.append(Transition(stateNum,act,endstate,0,(1-p)/len(l)))
				else:
					TransitionList.append(Transition(stateNum,act,data[i+elem[0]][j+elem[1]],-1,(1-p)/len(l)))


	for i in range(1,len(data)-1):
		for j in range(1,len(data[i])-1):
			if(data[i][j]!='1'):
				l=[]
				if(data[i-1][j]!='1'):
					l.append([-1,0])	
				if(data[i][j+1]!='1'):
					l.append([0,1])
				if(data[i+1][j]!='1'):
					l.append([1,0])
				if(data[i][j-1]!='1'):
					l.append([0,-1])


				if(data[i-1][j]!='1'):
					fillTransition(i,j,l,[-1,0],0,stateNum)	
				if(data[i][j+1]!='1'):
					fillTransition(i,j,l,[0,1],1,stateNum)
				if(data[i+1][j]!='1'):
					fillTransition(i,j,l,[1,0],2,stateNum)
				if(data[i][j-1]!='1'):
					fillTransition(i,j,l,[0,-1],3,stateNum)

				stateNum = stateNum+1
	gamma = 0.99		

	return stateNum,4,startstate,endstate,TransitionList,gamma



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-maze', help='maze description',dest ="maze",default='../base/data/maze/grid10.txt')
	parser.add_argument('-p', help='probability',dest ="p",default=1)

	args = parser.parse_args()
	mazePath = args.maze
	p = float(args.p)

	numstates,numactions,startstate,endstate,TransitionList,gamma = createMdp(mazePath,p)

	print("numStates",numstates)
	print("numActions",numactions)
	print("start",startstate)
	print("end",endstate)
	for t in (TransitionList):
		print("transition",t.state,t.action,t.next_state,t.reward,t.probability)
	print("discount","", gamma)





