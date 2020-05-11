#!/bin/bash

for (( i = 0; i < 10; i++ )); do
	nohup python3 train_cuda.py --start $i --end $[i+1] > ../data/log/log$i.txt 2>&1 &
	# python3 train_cuda1.py
done