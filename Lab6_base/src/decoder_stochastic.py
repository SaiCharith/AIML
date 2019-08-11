import argparse
import time
import random


def getroute(mazePath,policyPath,p):
	data = []
	with open(mazePath) as inputfile:
	    for line in inputfile:
	        data.append(line.strip().split(' '))

	Policydata = []
	with open(policyPath) as inputfile:
	    for line in inputfile:
	        Policydata.append(line.strip().split(' '))

	Data = [['1' for _ in range(len(data[0])+2)] for _ in range(len(data)+2)]
	for i in range(len(data)):
		for j in range(len(data[0])):
			Data[i+1][j+1] = data[i][j] 
	data = Data

	stateNum = 0
	endstate = -1
	startstate = -1
	states = []
	for i in range(1,len(data)-1):
		for j in range(1,len(data[i])-1):
			if(data[i][j]!='1'):
				if(data[i][j]=='3'):
					endstate = stateNum
				if(data[i][j]=='2'):
					startstate = stateNum
				data[i][j] = stateNum
				states.append([i,j])
				stateNum = stateNum + 1


	currState = startstate
	path =[]

	def getdir(currState,step,i,j):
		l=[]
		# print(data[i][j],data[i-1][j],data[i][j-1],data[i+1][j],data[i][j+1])
		if(data[i-1][j]!='1'):
			l.append(0)	
		if(data[i][j+1]!='1'):
			l.append(1)
		if(data[i+1][j]!='1'):
			l.append(2)
		if(data[i][j-1]!='1'):
			l.append(3)
		c = random.random()
		# print(l)
		if(c<=p):
			return step
		else:
			
			return (random.sample(l,1)[0])


	# print(currState,type(currState))
	while currState!=endstate:

		# print(currState)
		step = int(Policydata[currState][1])
		step = getdir(currState,step,states[currState][0],states[currState][1])
		if step==0:
			path.append('N')
			currState = data[states[currState][0]-1][states[currState][1]] 
		if step==1:
			path.append('E')
			currState = data[states[currState][0]][states[currState][1]+1] 
		if step==2:
			path.append('S')
			currState = data[states[currState][0]+1][states[currState][1]] 
		if step==3:
			path.append('W')
			currState = data[states[currState][0]][states[currState][1]-1] 
	return path



if __name__=='__main__':
	# sd = int(time.time()%97)
	# print(sd)
	random.seed(0)
	parser = argparse.ArgumentParser()
	parser.add_argument('-maze', help='maze description',dest ="maze",default='../base/data/maze/grid10.txt')
	parser.add_argument('-policy', help='policy description',dest ="policyPath",default='')
	parser.add_argument('-p', help='probability',dest ="p",default=1)

	args = parser.parse_args()
	mazePath = args.maze
	policyPath = args.policyPath
	p = float(args.p)

	path = getroute(mazePath,policyPath,p)

	s = ""
	for t in (path):
		s = (s + t )+" "

	print(s)





