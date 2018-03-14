"""train ddpg agent in initial environment
average rewards from 10-time training
"""
from __future__ import division
from multi_process import MultiProcess
import time
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow 

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

avg_rwd = []
done_count = 0
data_dir_list = os.listdir(rewards_dir)
for data_dir in data_dir_list:
	data_dir = os.path.join(rewards_dir, data_dir)
	file_name = os.path.join(data_dir, 'rwd.dat')
	if os.path.exists(file_name):
		file = open(file_name, 'rb')
		rwd = pickle.load(file)
		if avg_rwd == []:
			avg_rwd = np.asarray(rwd)
		else:
			avg_rwd += np.asarray(rwd)
		plt.plot(rwd)
	file = open(os.path.join(data_dir, 'done.dat'), 'rb')
	done = pickle.load(file)
	if done == 1:
		done_count += 1
avg_rwd /= len(data_dir_list)
x, y = (len(avg_rwd) - len(avg_rwd) / 3, max(avg_rwd))
plt.text(x, y, 'done rate: %d/%d' % (done_count, TRAIN_NUM))
plt.plot(avg_rwd, linewidth=2)
plt.savefig(os.path.join(rewards_dir, 'rwd_fig.png'))