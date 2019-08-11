

import argparse
import time
import os

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-grid', help='maze description',dest ="grid",default='10')
	parser.add_argument('-v', help='probability',dest ="v",default='0')
	parser.add_argument('-p', help='probability',dest ="p",default='1')

	args = parser.parse_args()
	p = (args.p)
	grid = (args.grid)
	v = args.v
	os.system("bash encoder.sh ../base/data/maze/grid"+ grid +".txt "+p+ "  > mdp")
	os.system("bash valueiteration.sh mdp > policy")
	os.system("bash decoder.sh ../base/data/maze/grid"+ grid +".txt policy "+p+"  > path")

	if v=='1':
		os.system('python3 ../base/visualize.py ../base/data/maze/grid'+ grid +'.txt path')
	







