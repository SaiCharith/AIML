#!/bin/bash
rm boosting_train.csv boosting_val.csv boosting_test.csv

for i in 1 3 5 7 9 10 13 15 17 20 22 24 26 28 30 32 34 36 40 50
do
	python2 dataClassifier.py -c boosting -t 1000 -s 1000 -b $i	
done
