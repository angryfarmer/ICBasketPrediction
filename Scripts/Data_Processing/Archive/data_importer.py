import numpy as np
import h5py
	
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
train_DSPO_set = "train_DSPO"
val_DSPO_set = "val_DSPO"
test_DSPO_set = "test_DSPO"
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
		self.train_DSPO = h5py.File(data_file_path,'r')[train_DSPO_set]
		self.val_DSPO = h5py.File(data_file_path,'r')[val_DSPO_set]
		self.test_DSPO = h5py.File(data_file_path,'r')[test_DSPO_set]
		self.current_load_index = 0
		self.current_train_index = 0
		self.load_batch = load_batch
		self.train_batch = train_batch
		self.next_load_batch()
		self.end_of_file = False
	def next_load_batch(self):
			# print(np.shape(self.val_users))
			self.batch_val_users = self.val_users[self.current_load_index:self.current_load_index+self.load_batch]
			self.batch_test_users = self.test_users[self.current_load_index:self.current_load_index+self.load_batch]
			self.current_load_batch = self.h5f_train[self.current_load_index:self.current_load_index+self.load_batch,:,:,0:1]
			self.current_DSPO_batch = self.train_DSPO[self.current_load_index:self.current_load_index+self.load_batch]
			self.batch_val_DSPO = self.val_DSPO[self.current_load_index:self.current_load_index+self.load_batch]
			self.batch_test_DSPO = self.test_DSPO[self.current_load_index:self.current_load_index+self.load_batch]

			# print("Shape: {}".format(np.shape(self.h5f_val)))
			if(self.include_val):
				for n in range(np.shape(self.current_load_batch)[0]):
					if(self.batch_val_users[n] > 0):
						# print("Batch: {},{}, Current Load Batch Shape: {}, Current Val Shape: {}".format(n,self.current_load_index+n,np.shape(self.current_load_batch[n:n+1,1:,:,:]),np.shape(self.h5f_val[self.current_load_index+n:self.current_load_index+n+1,:,:,:])))

						self.current_load_batch[n:n+1,:,:,:] = np.concatenate((self.current_load_batch[n:n+1,1:,:,:],self.h5f_val[self.current_load_index+n:self.current_load_index+n+1,:,:,:]),axis = 1)
						# print("{}, {}".format(np.shape(self.current_DSPO_batch[n:n+1,1:]),np.shape(self.batch_val_DSPO[self.current_load_index+n:self.current_load_index+n+1,:])))
						self.current_DSPO_batch[n:n+1,:] = np.concatenate((self.current_DSPO_batch[n:n+1,1:],self.batch_val_DSPO[n:n+1,:]),axis = 1)
			if(self.include_test):
				for n in range(np.shape(self.current_load_batch)[0]):
					if(self.batch_test_users[n] > 0):	
						# print(self.current_load_index+n)
						self.current_load_batch[n:n+1,:,:,:] = np.concatenate((self.current_load_batch[n:n+1,1:,:,:],self.h5f_test[self.current_load_index+n:self.current_load_index+n+1,:,:,:]),axis = 1)
						self.current_DSPO_batch[n:n+1,:] = np.concatenate((self.current_DSPO_batch[n:n+1,1:],self.batch_test_DSPO[n:n+1,:]),axis = 1)
				
			self.current_train_index = 0
	def next_training_sample(self):
		current_train_sample = np.array([])
		current_DSPO_sample = np.array([])
		for n in range(self.train_batch):
			if(self.include_test):
				if(self.batch_test_users[self.current_train_index+n:self.current_train_index+n+1] > 0):
					if(np.size(current_train_sample) == 0):
						current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
						current_DSPO_sample = self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]
					else:
						current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
						current_DSPO_sample = np.concatenate((current_DSPO_sample,self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
			else:
				if(self.include_val):
					if(self.batch_val_users[self.current_train_index+n:self.current_train_index+n+1] > 0):
						if(np.size(current_train_sample) == 0):

							current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
							current_DSPO_sample = self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]
						else:
							current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
							current_DSPO_sample = np.concatenate((current_DSPO_sample,self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
				else:
					if(np.size(current_train_sample) == 0):
						current_train_sample = self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]
						current_DSPO_sample = self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]
					else:
						current_train_sample = np.concatenate((current_train_sample,self.current_load_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
						current_DSPO_sample = np.concatenate((current_DSPO_sample,self.current_DSPO_batch[self.current_train_index+n:self.current_train_index+n+1]),axis = 0)
		
		self.current_train_index += self.train_batch
		self.current_load_index += self.train_batch
		if(self.current_train_index >= self.load_batch):
			self.next_load_batch()
		if(self.current_load_index >= self.h5f_train.shape[0]):
			self.end_of_file = True
		return current_train_sample, current_DSPO_sample
	def reset_to_head(self):
		self.current_load_index = 0
		self.next_load_batch()
		self.end_of_file = False

test = False
test2 = False
test3 = False
test3 = True
# test2 = True
# test = True

if(test3):
	data_file_path = "..\\..\\Processed_Data\\user_baskets"
	data = data_importer(data_file_path,load_batch = 30,include_val = True)
	user_indices = np.reshape(np.nonzero(data.val_users),-1)
	# print(user_indices)
	# print(np.sum(data.h5f_train))
	prior_items = np.amax(data.h5f_train,axis = (3,1))
	prior_index = np.nonzero(prior_items)
	test_items 	= np.amax(data.h5f_val,axis = (3,1))
	test_index = np.nonzero(test_items)
	mapping = np.multiply(prior_items,test_items)
	# print(prior_index)
	# print(test_index)
	val_user_items = 0
	for n in range(np.shape(user_indices)[0]):
		# val_user_items += np.sum(prior_items[user_indices[n],:])
		# print(prior_index[])
		print(user_indices[n])
		print(np.sum(prior_items[user_indices[n],:]))
		print(np.sum(test_items[user_indices[n],:]))
	# print(val_user_items)
	# print(np.sum(prior_items))
	# print(np.sum(test_items))

if(test2):
	data_file_path = "..\\..\\Processed_Data\\user_baskets"
	data = data_importer(data_file_path,load_batch = 30,include_val = True)
	# print(np.sum(data.h5f_val[:,:,:,0]))
	user_indices = np.reshape(np.nonzero(data.val_users),-1)
	tot = 0
	tot2 = 0
	# print(np.shape(user_indices))
	for n in range(np.shape(user_indices)[0]):
		# print(user_indices[n])
		temp = data.h5f_train[int(user_indices[n]):int(user_indices[n]+1),:,:,0]
		if(np.isnan(np.sum(temp))):
			print("N: {}, SUM: {}".format(n,np.sum(temp)))
	# 	tot2 += np.sum(temp)
	# 	tot += np.sum(np.amax(temp,axis = 1))
	# print(tot)
	# print(tot2)

	# tot = 0
	# while(not data.end_of_file):
	# 	tot += np.sum(np.amax(data.next_training_sample()[:,:-1,:,0],axis = 1))
	# 	print(tot)

	# print(tot)
	# print(np.sum(np.amax(data.h5f_train[1:,:,:,0],axis = 1)))

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





