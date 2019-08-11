#!/bin/bash
foo=${2:-1}
python3 encoder_stochastic.py -maze $1 -p $foo 
