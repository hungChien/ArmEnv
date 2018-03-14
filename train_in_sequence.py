import time
import os
from multi_process import MultiProcess
import pickle
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 150
TRAIN_NUMS = 4
TRAIN_ORIGINAL_MODEL = False

procs = MultiProcess()
start_time = time.strftime('%m%d-%H%M%S')
start = time.time()

rewards_dir = os.path.join('./results/rewards/', start_time, 'old')

for i in range(TRAIN_NUMS):
	pass
	procs.add_one_process(['python3','ddpg3_v2_pnn.py','--max-episodes',
		str(MAX_EPISODES),'--rewards-dir',rewards_dir,'--cascade-net'])
while True:
	if procs.all_done():
		break
	time.sleep(5)
# plot old rewards
rwd_lists = []
rewards_dir_old = rewards_dir
for data_dir in os.listdir(rewards_dir_old):
	file_name = os.path.join(rewards_dir_old, data_dir, 'rwd.dat')
	reward_file = open(file_name, 'rb')
	rwd_lists.append(np.array(pickle.load(reward_file)))
average_rwd = []
for lst in rwd_lists:
	plt.plot(lst,'b--')
	if average_rwd == []:
		average_rwd = lst
	else:
		average_rwd += lst
average_rwd /= len(rwd_lists)
plt.plot(average_rwd,'r', linewidth = 2)



rewards_dir = os.path.join('./results/rewards/', start_time, 'new')
for i in range(TRAIN_NUMS):
	procs.add_one_process(['python3','ddpg3_v2_pnn.py','--max-episodes',
		str(MAX_EPISODES),'--rewards-dir',rewards_dir])
while True:
	if procs.all_done():
		break
	time.sleep(5)
# plot new rewards
rwd_lists = []
rewards_dir_new = rewards_dir
for data_dir in os.listdir(rewards_dir_new):
	file_name = os.path.join(rewards_dir_new, data_dir, 'rwd.dat')
	reward_file = open(file_name, 'rb')
	rwd_lists.append(np.array(pickle.load(reward_file)))
average_rwd = []
for lst in rwd_lists:
	plt.plot(lst,'b--')
	if  average_rwd == []:
		average_rwd = lst
	else:
		average_rwd += lst
average_rwd /= len(rwd_lists)
plt.plot(average_rwd,'g', linewidth = 2)
fig_name = os.path.join(rewards_dir, 'rwd_fig.png')
plt.savefig(fig_name)

elapsed = time.time() - start
elapsed //= 60
print('time elapsed: %d min' % elapsed)


# os.system('python3 ddpg3.py --max-episodes 5 --save-model')
# os.system('python3 ddpg3_pnn.py --max-episodes 5')