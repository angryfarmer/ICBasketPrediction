import numpy as np
import h5py
	
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"

class data_importer():
	def __init__(self,data_file_path,load_batch = 1,train_batch = 1,include_val = False,include_test = False):
		self.include_val = include_val
		self.include_test = include_test
		if(include_test):
			self.include_val = True
		self.h5f_train = h5py.File(data_file_path,'r')[train_set_name]
		self.h5f_val = h5py.File(data_file_path,'r')[val_set_name]
		self.h5f_test = h5py.File(data_file_path,'r')[test_set_name]
		self.current_load_index = 0
		self.current_train_index = 0
		self.load_batch = load_batch
		self.train_batch = train_batch
		self.next_load_batch()
		self.end_of_file = False
	def next_load_batch(self):
			self.current_load_batch = self.h5f_train[self.current_load_index:self.current_load_index+self.load_batch]
			if(self.include_val):
				self.current_load_batch = np.concatenate((self.current_load_batch[:,1:,:,:],self.h5f_val[self.current_load_index:self.current_load_index+self.load_batch]),axis = 1)
			if(self.include_test):
				self.current_load_batch = np.concatenate((self.current_load_batch[:,1:,:,:],self.h5f_test[self.current_load_index:self.current_load_index+self.load_batch]),axis = 1)
				
			self.current_train_index = 0
	def next_training_sample(self):
		current_train_sample = self.current_load_batch[self.current_train_index:self.current_train_index+self.train_batch]
		self.current_train_index += self.train_batch
		self.current_load_index += self.train_batch
		if(self.current_train_index >= self.load_batch):
			self.next_load_batch()
		if(self.current_load_index >= self.h5f_train.shape[0]):
			self.end_of_file = True
		return current_train_sample
	def reset_to_head(self):
		self.current_load_index = 0
		self.next_load_batch()
		self.end_of_file = False


test = True
if(test):	
	data_file_path = "..\\..\\Processed_Data\\user_baskets"

	data = data_importer(data_file_path,load_batch = 30)
	n = 0
	print(data.h5f_test.shape)
	for k in range(1):
		n = 0
		while(not data.end_of_file):
			print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
			n += 1
		data.reset_to_head()

	data = data_importer(data_file_path,load_batch = 30,include_val = True)
	n = 0
	print(data.h5f_test.shape)
	for k in range(1):
		n = 0
		while(not data.end_of_file):
			print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
			n += 1
		data.reset_to_head()

	data = data_importer(data_file_path,load_batch = 30,include_test = True)
	n = 0
	print(data.h5f_test.shape)
	for k in range(1):
		n = 0
		while(not data.end_of_file):
			print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
			n += 1
		data.reset_to_head()
