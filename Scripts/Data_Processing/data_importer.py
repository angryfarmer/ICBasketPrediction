import numpy as np
import h5py
	
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
val_user_set_name = "val_users"
test_user_set_name = "test_users"

class data_importer():
	def __init__(self,data_file_path,load_batch = 1,train_batch = 1,include_val = False,include_test = False):
		self.include_val = include_val
		self.include_test = include_test
		if(include_test):
			self.include_val = True
		self.h5f_train = h5py.File(data_file_path,'r')[train_set_name]
		self.h5f_val = h5py.File(data_file_path,'r')[val_set_name]
		self.h5f_test = h5py.File(data_file_path,'r')[test_set_name]
		self.val_users = h5py.File(data_file_path,'r')[val_user_set_name]
		self.test_users = h5py.File(data_file_path,'r')[test_user_set_name]
		self.current_load_index = 0
		self.current_train_index = 0
		self.load_batch = load_batch
		self.train_batch = train_batch
		self.next_load_batch()
		self.end_of_file = False
	def next_load_batch(self):
			self.batch_val_users = self.val_users[self.current_load_index:self.current_load_index+self.load_batch]
			self.batch_test_users = self.test_users[self.current_load_index:self.current_load_index+self.load_batch]
			self.current_load_batch = self.h5f_train[self.current_load_index:self.current_load_index+self.load_batch]
			# print("Shape: {}".format(np.shape(self.h5f_val)))
			if(self.include_val):
				for n in range(np.shape(self.current_load_batch)[0]):
					if(self.batch_val_users[n] > 0):
						# print("Batch: {},{}, Current Load Batch Shape: {}, Current Val Shape: {}".format(n,self.current_load_index+n,np.shape(self.current_load_batch[n:n+1,1:,:,:]),np.shape(self.h5f_val[self.current_load_index+n:self.current_load_index+n+1,:,:,:])))

						self.current_load_batch[n:n+1,:,:,:] = np.concatenate((self.current_load_batch[n:n+1,1:,:,:],self.h5f_val[self.current_load_index+n:self.current_load_index+n+1,:,:,:]),axis = 1)
			if(self.include_test):
				for n in range(np.shape(self.current_load_batch)[0]):
					if(self.batch_test_users[n] > 0):	
						# print(self.current_load_index+n)
						self.current_load_batch[n:n+1,:,:,:] = np.concatenate((self.current_load_batch[n:n+1,1:,:,:],self.h5f_test[self.current_load_index+n:self.current_load_index+n+1,:,:,:]),axis = 1)
				
			self.current_train_index = 0
	def next_training_sample(self):
		current_train_sample = np.array([])
		for n in range(self.train_batch):
			if(self.include_test):
				if(self.batch_test_users[self.current_train_index+n:self.current_train_index+n+1] > 0):
					if(current_train_sample.size == 0):
						current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
					else:
						current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
			else:
				if(self.include_val):
					if(self.batch_val_users[self.current_train_index+n:self.current_train_index+n+1] > 0):
						if(current_train_sample.size == 0):
							current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
						else:
							current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
				else:
					if(current_train_sample.size == 0):
						current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
					else:
						current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
		
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

	# data = data_importer(data_file_path,load_batch = 30)
	# n = 0
	# print(data.h5f_test.shape)
	# for k in range(1):
	# 	n = 0
	# 	while(not data.end_of_file):
	# 		print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
	# 		n += 1
	# 	data.reset_to_head()

	# data = data_importer(data_file_path,load_batch = 30,include_val = True)
	# n = 0
	# print(data.h5f_test.shape)
	# for k in range(1):
	# 	n = 0
	# 	while(not data.end_of_file):
	# 		print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
	# 		n += 1
	# 	data.reset_to_head()

	data = data_importer(data_file_path,load_batch = 30,include_test = True)
	n = 0
	print(data.h5f_train.shape)
	print(data.h5f_val.shape)
	print(data.h5f_test.shape)
	# for k in range(1):
	# 	n = 0
	# 	while(not data.end_of_file):
	# 		print("Step: {}, Sum:{}".format(n,np.sum(data.next_training_sample())))	
	# 		n += 1
	# 	data.reset_to_head()
	n = 0
	while(not data.end_of_file):
		sample = data.next_training_sample()
		val_ok = True
		test_ok = True
		train_ok = True
		train_sum = 0
		sample_sum = 0
		if(data.test_users[n] > 0 and sample.size == 0):
			print("n: {}".format(n))
		if(int(data.test_users[n]) > 0):
			test_ok = np.all(sample[:,-1,:,:] == data.h5f_test[n:n+1,0,:,:])
			if(data.val_users[n] > 0):
				val_ok = np.all(sample[:,-2,:,:] == data.h5f_val[n:n+1,0,:,:])
				train_ok = np.all(sample[:,:-2,:,:] == data.h5f_train[n:n+1,2:,:,:])
				sample_sum = np.sum(sample[:,:-2,:,:])
				train_sum = np.sum(data.h5f_train[n:n+1,2:,:,:])
			else:
				sample_sum = np.sum(sample[:,:-2,:,:])
				train_sum = np.sum(data.h5f_train[n:n+1,2:,:,:]) 	 	
				train_ok = np.all(sample[:,:-1,:,:] == data.h5f_train[n:n+1,1:,:,:])
		if(not (train_ok and val_ok and test_ok)):
			print("n: {},Val: Users: {}, Train: {}, Val: {}, Test: {}, Train Sum: {}, Sample Sum: {}".format(n,data.val_users[n],train_ok,val_ok,test_ok,train_sum,sample_sum))
			# print("Raw: {}, Sample: {}".format(np.sum(data.h5f_test[n:n+1,0,:,:]),np.sum(sample)))

		n += 1
		# print(sample)
		# val_ok = True

		# test_ok = True
		# train_ok = True
		# if(int(data.test_users[n]) > 0):
		# 	test_ok = np.all(sample[:,-1,:,:] == data.h5f_test[n:n+1,0,:,:])
		# 	if(data.val_users[n] > 0):
		# 		val_ok = np.all(sample[:,-2,:,:] == data.h5f_val[n:n+1,0,:,:])
		# 		train_ok = np.all(sample[:,:-2,:,:] == data.h5f_train[n:n+1,2:,:,:])
		# 	else:
		# 		train_ok = np.all(sample[:,:-1,:,:] == data.h5f_train[n:n+1,1:,:,:])
		# if(int(data.test_users[n]) == 0 and data.val_users[n] > 0):
		# 	val_ok = np.all(sample[:,-1,:,:] == data.h5f_val[n:n+1,0,:,:])
		# 	train_ok = np.all(sample[:,:-1,:,:] == data.h5f_train[n:n+1,1:,:,:])
		# if(int(data.test_users[n]) == 0 and int(data.val_users[n]) == 0):
		# 	train_ok = np.all(sample[:,:,:,:] == data.h5f_train[n:n+1,:,:,:])
		# if(not (train_ok and val_ok and test_ok)):
		# 	print("n: {}, Train: {}, Val: {}, Test: {}".format(n,train_ok,val_ok,test_ok))
		# n += 1





