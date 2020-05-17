#!/bin/bash

for (( i = 0; i < 2; i++ )); do
	nohup python3 train_cuda.py --start $i --end $[i+1] --device 0 > ../data/log_gcn/log$i.txt 2>&1 &
done

for (( i = 2; i < 4; i++ )); do
	nohup python3 train_cuda.py --start $i --end $[i+1] --device 1 > ../data/log_gcn/log$i.txt 2>&1 &
done

for (( i = 4; i < 7; i++ )); do
	nohup python3 train_cuda.py --start $i --end $[i+1] --device 2 > ../data/log_gcn/log$i.txt 2>&1 &
done

for (( i = 7; i < 10; i++ )); do
	nohup python3 train_cuda.py --start $i --end $[i+1] --device 3 > ../data/log_gcn/log$i.txt 2>&1 &
done