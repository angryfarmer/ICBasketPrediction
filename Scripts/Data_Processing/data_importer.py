import numpy as np
import h5py
	
dataset_name = "rnn_data_input"

class data_importer():
	def __init__(self,data_file_path,load_batch = 1,train_batch = 1):
		self.h5f = h5py.File(data_file_path,'r')[dataset_name]
		self.current_load_index = 0
		self.current_train_index = 0
		self.load_batch = load_batch
		self.train_batch = train_batch
		self.next_load_batch()
		self.end_of_file = False
	def next_load_batch(self):
			self.current_load_batch = self.h5f[self.current_load_index:self.current_load_index+self.load_batch]
			self.current_train_index = 0
	def next_training_sample(self):
		current_train_sample = self.current_load_batch[self.current_train_index:self.current_train_index+self.train_batch]
		self.current_train_index += self.train_batch
		self.current_load_index += self.train_batch
		if(self.current_train_index >= self.load_batch):
			self.next_load_batch()
		if(self.current_load_index >= self.h5f.shape[0]):
			self.end_of_file = True
		return current_train_sample
	def reset_to_head(self):
		self.current_load_index = 0
		self.next_load_batch()
		self.end_of_file = False


test = False
if(test):
	data_file_path = "..\\Processed_Data\\user_baskets"

	data = data_importer(data_file_path,load_batch = 30)
	n = 0
	eof = False
	print(data.h5f.shape)
	for k in range(1):
		n = 0
		while(not data.end_of_file):
			print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
			n += 1
		data.reset_to_head()