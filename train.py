from multi_process import MultiProcess
import time
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

MAX_EPISODES = 100
TRAIN_NUM = 10

start = time.time()
start_time = time.strftime('%m%d-%H%M%S')
rewards_dir = os.path.join('./results/rewards', start_time)
procs = MultiProcess(max_proc_num = 5)
for _ in range(TRAIN_NUM):
	cmd = ['python', 'ddpg3_v2.py', '--max-episodes', str(MAX_EPISODES),
				'--rewards-dir', rewards_dir]
	procs.add_one_process(cmd)
while True:
	if procs.all_done():
		break
elapsed = time.time() - start
elapsed //= 60
print('time elapsed %d min' % elapsed)