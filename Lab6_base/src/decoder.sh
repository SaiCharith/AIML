#!/bin/bash
foo=${3:-1}
python3 decoder_stochastic.py -maze $1 -policy $2 -p $foo 
