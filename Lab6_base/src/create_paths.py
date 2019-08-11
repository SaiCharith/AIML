

import os
import matplotlib.pyplot as plt


x=[]
y=[]
J = list(range(0,21))
for i in range(len(J)):
	j = J[i]/20.0
	os.system("bash encoder.sh ../base/data/maze/grid10.txt "+ str(j) + " > mdp10 ")
	os.system("bash valueiteration.sh mdp10 > policy")
	os.system("bash decoder.sh ../base/data/maze/grid10.txt policy "+ str(j) + " > path" )
	data = []
	print(j)
	with open("path") as inputfile:
		for line in inputfile:
	   		data = line.strip().split(' ')
	   		break
	print(len(data))
	print(data)
	x.append(j)
	y.append(len(data))
plt.plot(x,y)
plt.xlabel("p")
plt.ylabel("Expected no. of steps")
plt.title("p vs path_length")
plt.savefig("plot.png")
plt.show()

