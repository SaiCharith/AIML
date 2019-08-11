#!/bin/bash
rm bagging_train.csv bagging_val.csv bagging_test.csv

for i in 1 3 5 7 9 10 13 15 17 20 22 24 26 28 30 32 34 36 40 50
do
	python2 dataClassifier.py -c bagging -t 1000 -s 1000 -r 1 -n $i
done
