import numpy as np
import h5py
import time
import os
	
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
train_DSPO_set = "train_DSPO"
val_DSPO_set = "val_DSPO"
test_DSPO_set = "test_DSPO"
val_user_set_name = "val_users"
test_user_set_name = "test_users"
set_information = "set_info"


class data_importer():
	def __init__(self,data_file_path,load_batch = 1,train_batch = 1,include_val = False,include_test = False):
		## Load data sets
		self.h5f_train = h5py.File(data_file_path,'r')[train_set_name][:,:]
		self.h5f_val = h5py.File(data_file_path,'r')[val_set_name][:,:]
		self.h5f_test = h5py.File(data_file_path,'r')[test_set_name][:,:]
		
		## Data shape information
		self.set_information = h5py.File(data_file_path,'r')[set_information]
		self.time_steps = int(self.set_information[1])
		# print(self.time_steps)
		self.products = int(self.set_information[2])
		self.train_user_steps = np.bincount(self.h5f_train[:,0].astype(np.int64))
		self.val_user_steps = np.bincount(self.h5f_val[:,0].astype(np.int64))
		self.test_user_steps = np.bincount(self.h5f_test[:,0].astype(np.int64))

		## Looping variables
		self.end_of_file = False
		self.index = 0
		self.train_batch = train_batch

		## Prepare data based on eval set
		self.include_val = include_val
		self.include_test = include_test
		self.user_list = np.unique(self.h5f_train[:,0])
		if(include_val):
			self.user_list = np.unique(self.h5f_val[:,0])

		if(include_test):
			self.include_val = True
			self.user_list = np.unique(self.h5f_test[:,0])

	def next_training_sample(self):
		
		batch_data = np.array([])
		limit = self.index + self.train_batch
		users = np.array(self.user_list[self.index:limit])
		sequence_length = np.zeros(np.shape(users)[0])
		batch_user_index = 0

		## Initialize arrays for single user
		# start = time.time()
		users_in_sample = min([len(self.user_list) - self.index,self.train_batch]) 
		sample = np.zeros([users_in_sample,self.time_steps,self.products,1])
		sample_dspo = np.zeros([users_in_sample,self.time_steps])
		# print("Initialize users time: {}".format(time.time()-start))
		while self.index < limit and self.index < len(self.user_list):
			## Load data for specific user. user_id is 1-index based
			# start = time.time()
			user_id = self.user_list[self.index]
			user_data = self.user_data_points(user_id)
			# print("Get user indicies  time: {}".format(time.time()-start))
			
			## Get number of orders in order to know how many zeros to pad data.
			user_number_of_orders = np.amax(user_data[:,1])
			sequence_length[batch_user_index] = user_number_of_orders - 1
			
			## Loop through indices and add data points to orders
			for n in range(np.shape(user_data)[0]):
				## All values start with index 1
				user_order_number = int(user_data[n,1])
				dspo = int(user_data[n,3])
				if(int(user_data[n,1]) == 1):
					dspo = 0
				product_id = int(user_data[n,2])

				## Add data to arrays
				sample_dspo[batch_user_index,int(user_order_number - 1 - 1)] = dspo
				sample[batch_user_index,int(user_order_number - 1),product_id - 1,0] = 1
			
			## Add DSPO for test set. Test set should only have 1 DSPO per user so loop not required
			if(self.include_test):
				start_index = np.sum(self.train_user_steps[:user_id])
				end_index = np.sum(self.train_user_steps[:user_id+1])
				dspo = self.h5f_test[start_index:end_index]
				sample_dspo[batch_user_index,(dspo[0,1]-2)] = dspo[0,0]
				sequence_length[batch_user_index] += 1

			self.index += 1
			batch_user_index += 1

		## If we've iterated through all users, mark end of file as true 
		if(self.index >= len(self.user_list)):
			self.end_of_file = True
		return sample,sample_dspo,users,sequence_length
	
	def user_data_points(self,user_id):
		start_index = np.sum(self.train_user_steps[:user_id])
		end_index = np.sum(self.train_user_steps[:user_id+1])

		user_data = self.h5f_train[start_index:end_index]
		# if(len(np.unique(user_data[:,0])) > 1):
		# 	print("Problem with unique users")
		# 	print(end_index - start_index)
		# 	print(np.unique(user_data[:,0]))
		if(self.include_val):
			start_index = np.sum(self.val_user_steps[:user_id])
			end_index = np.sum(self.val_user_steps[:user_id+1])
			user_data = np.concatenate((user_data,self.h5f_val[start_index:end_index]),axis = 0)
		return user_data

	def reset_to_head(self):
		self.index = 0
		self.end_of_file = False

test = False
# test = True
if(test):
	data_file_path = '..'+os.sep+'..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc'
	data2 = data_importer(data_file_path,train_batch = 10)
	data = data_importer(data_file_path,include_val = True,train_batch = 10)
	n = 0
	# print(data.user_data_points(1636))
	while not data.end_of_file:
		start = time.time()
		a,b,c,d = data.next_training_sample()
		# print(time.time() - start)
		n += 1
		if(n % 10 == 0):
		# if(False):
			print(np.sum(a[:,:20,:,:]))
			print(np.sum(a[:,20:40,:,:]))
			print(np.sum(a[:,40:60,:,:]))
			print(np.sum(a[:,60:80,:,:]))
			print(np.sum(a[:,80:100,:,:]))
			# print(np.sum(a[:,-2:-1,:,:]))
			print(np.shape(a))
			print(np.max(d))
			# userlen = data.train_user_steps[c[0]]
			# print(c[0])
			# print(np.sum(a[0,np.shape(a)[1]-userlen:,:,:]))
			# print(np.sum(a[0,:np.shape(a)[1]-userlen]))
			print("User: {}, Data Shape: {}, DSPO Shape: {}, Data Sum: {}, DSPO Sum: {}, Time Elapsed: {:0.2f}".format(c[0],np.shape(a),np.shape(b),np.sum(a),np.sum(b),time.time()-start))
	# print(n)