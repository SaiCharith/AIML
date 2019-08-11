import argparse
import time
from mdphelp import *


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-mdp', help='MDP description',dest ="mdp",default='../base/data/mdp/mdpfile01.txt')

	args = parser.parse_args()
	mdpPath = args.mdp

	numstates,numactions,startstate,endstates,TransitionList,gamma = getTransitions(mdpPath)

	t = time.time()
	V,p,iters = findOptimal1(numstates,numactions,TransitionList,gamma)
	t = time.time()-t
	# print(V,p)
	for i in range(numstates):
		if(endstates[i]!=1):
			print(V[i],p[i])
		else:
			print(V[i],-1)

	print("iterations",iters)
	# print(t)




