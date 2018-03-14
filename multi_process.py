import time
import subprocess

class MultiProcess(object):
	def __init__(self, max_proc_num):
		self.max_proc_num = max_proc_num
		self.procs = []
		self.count = 0
	def add_one_process(self, cmd):
		self.count += 1
		if self.count > self.max_proc_num:
			self._delete_process()
		tmp = subprocess.Popen(cmd)
		self.procs.append(tmp)
	def _delete_process(self):
		while True:
			poll_res = []
			for proc in self.procs:
				poll_res.append(proc.poll() == 0)
			if(any(poll_res)):
				idx = poll_res.index(True) # index only returns the first index that satisfies the condition
				self.procs.pop(idx)
				self.count -= 1
				return
			time.sleep(1)
	def all_done(self):
		poll_res = []
		for proc in self.procs:
			poll_res.append(proc.poll() == 0)
		if(all(poll_res)):
			self.count = 0
			return True
		else:
			return False