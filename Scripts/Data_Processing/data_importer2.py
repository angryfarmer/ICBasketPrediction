import numpy as np
import h5py
import time
	
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
		
		## Output data shape information
		self.set_information = h5py.File(data_file_path,'r')[set_information]
		self.time_steps = int(self.set_information[1])
		print(self.time_steps)
		self.products = int(self.set_information[2])
		
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
		users = np.array([])
		while self.index < limit and self.index < len(self.user_list):
			## Initialize arrays for single user
			sample = np.zeros([1,self.time_steps,self.products,1])
			sample_dspo = np.zeros([1,self.time_steps])

			## Load data for specific user. user_id is 1-index based
			user_id = self.user_list[self.index]
			user_data = self.user_data_points(user_id)
			user_number_of_orders = np.amax(user_data[:,1])
			for n in range(np.shape(user_data)[0]):
				## All values start with index 1
				user_order_number = int(user_data[n,1])
				dspo = int(user_data[n,3])
				if(int(user_data[n,1]) == 1):
					dspo = 0
				# print("{}, {}".format(n,dspo))
				product_id = int(user_data[n,2])

				## Calculate timestep index. 0 index based
				time_step = self.time_step_index(user_order_number,user_id,user_number_of_orders)
				
				## Add data to arrays
				sample_dspo[0,time_step] = dspo
				sample[0,time_step,product_id - 1,0] = 1

			if(self.include_test):
				dspo = self.h5f_test[self.h5f_test[:,0] == user_id]
				print(np.shape(dspo))
				sample_dspo[0,-1] = dspo[0,0]
			## Override batch data if batch data is empty. Otherwise concatenate
			if(np.size(batch_data) == 0):
				batch_data = sample
				batch_dspo = sample_dspo
				users = np.array([user_id])
			else:
				batch_data = np.concatenate((batch_data,sample),axis = 0)
				batch_dspo = np.concatenate((batch_dspo,sample),axis = 0)
				users = np.concatenate((users,np.array([user_id])))
			self.index += 1
		## If we've iterated through all users, mark end of file as true 
		if(self.index >= len(self.user_list)):
			self.end_of_file = True
		return batch_data,batch_dspo,users
	def time_step_index(self,order_number,user_id,user_number_of_orders):
		user_number_of_orders = user_number_of_orders + int(self.include_test)
		return int(self.time_steps - 1 - (user_number_of_orders - 1) + order_number - 1)

	def user_data_points(self,user_id):
		user_data = self.h5f_train[self.h5f_train[:,0] == user_id]
		if(self.include_val):
			user_data = np.concatenate((user_data,self.h5f_val[self.h5f_val[:,0] == user_id]),axis = 0)
		return user_data

	def reset_to_head(self):
		self.index = 0
		self.end_of_file = False

test = False
# test = True
if(test):
	data_file_path = "..\\..\\Processed_Data\\user_baskets_loc"
	data = data_importer(data_file_path,include_val = True)
	n = 0
	while not data.end_of_file:
		start = time.time()
		a,b,c = data.next_training_sample()
		n += 1
		if(n % 100 == 0):
			print(np.sum(a[:,:20,:,:]))
			print(np.sum(a[:,20:40,:,:]))
			print(np.sum(a[:,40:60,:,:]))
			print(np.sum(a[:,60:80,:,:]))
			print(np.sum(a[:,80:100,:,:]))
			print("User: {}, Data Shape: {}, DSPO Shape: {}, Data Sum: {}, DSPO Sum: {}, Time Elapsed: {:0.2f}".format(c[0],np.shape(a),np.shape(b),np.sum(a),np.sum(b),time.time()-start))
	print(n)