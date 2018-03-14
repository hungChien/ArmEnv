from __future__ import division
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

start_time = '0314-165129'
rewards_dir = os.path.join('./results/rewards', start_time)
TRAIN_NUM = 10

avg_rwd = []
done_count = 0
data_dir_list = os.listdir(rewards_dir)
for data_dir in data_dir_list:
    data_dir = os.path.join(rewards_dir, data_dir)
    file = open(os.path.join(data_dir, 'rwd.dat'), 'rb')
    rwd = pickle.load(file)
    if avg_rwd == []:
        avg_rwd = np.asarray(rwd)
    else:
        avg_rwd += np.asarray(rwd)
    plt.plot(rwd)
    file = open(os.path.join(data_dir, 'done.dat'), 'rb')
    done = pickle.load(file)
    print(type(done))
    if done == 1:
        done_count += 1
avg_rwd /= len(data_dir_list)
print(done_count)
x, y = (len(avg_rwd) - len(avg_rwd) / 3, max(avg_rwd))
plt.text(x, y, 'done rate: %d / %d' % (done_count, TRAIN_NUM))
plt.plot(avg_rwd, linewidth=2)
plt.savefig(os.path.join(rewards_dir, 'rwd_fig.png'))