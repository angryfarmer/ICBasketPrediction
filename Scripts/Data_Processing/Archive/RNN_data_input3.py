import numpy as np
import pandas as pd
import os
import h5py
import time

Tst = False
Tst = True
product_features = 1
inpath_str = "..\\..\\Data\\"
data_file_path = "..\\..\\Processed_Data\\user_baskets"
val_set_path = "..\\..\\Processed_Data\\val_set"
test_set_path = "..\\..\\Processed_Data\\test_set"
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
train_DSPO_set = "train_DSPO"
val_DSPO_set = "val_DSPO"
test_DSPO_set = "test_DSPO"
val_user_set_name = "val_users"
test_user_set_name = "test_users"
train = "prior"
val = "train"
test = "test"

if(not Tst):
	##### Training Data #####
	# aisles_df               = pd.read_csv(inpath_str + 'aisles.csv')
	# departments_df          = pd.read_csv(inpath_str + 'departments.csv')
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
	# order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
	# order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
	# order_products_df 		= pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv')
	products_df             = pd.read_csv(inpath_str + 'products.csv')
else:
	##### Test Data #####
	# aisles_df               = pd.read_csv(inpath_str + 'aisles.csv',nrows = 1000)
	# departments_df          = pd.read_csv(inpath_str + 'departments.csv',nrows = 1000)
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
	# print(sum(1 for row in order_products_prior_df))
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
	order_products_prior_df_test = pd.read_csv(inpath_str + 'order_products__prior.csv',nrows = 500000)
	# order_products_prior_df_test = pd.concat(order_products_prior_df_test,ignore_index = True)
	order_products_train_df_test = pd.read_csv(inpath_str + 'order_products__train.csv',nrows = 500000)
	# order_products_train_df_test = pd.concat(order_products_train_df_test, ignore_index = True)

	order_products_df_test 		= pd.concat([order_products_prior_df_test,order_products_train_df_test],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv')
	print(len(orders_df))
	orders_df 				= orders_df[orders_df.user_id < 200]
	products_df             = pd.read_csv(inpath_str + 'products.csv')

# temp_df = pd.merge(orders_df[(orders_df.eval_set == 'prior')],order_products_df,on = 'order_id',how = 'left')
# temp_df = temp_df[temp_df.product_id.notnull()]
# print(temp_df[(temp_df.user_id == 54) & (temp_df.order_number == 29)].head(30))

## Orders DF setup
order_types 			= orders_df.eval_set.unique()
print(order_types)
orders_df 				= orders_df.sort_values('user_id')

## Script Constants
user_number_of_orders 	= orders_df.groupby(['user_id']).size()
time_steps 				= user_number_of_orders.max()
number_of_products 		= len(products_df) 
number_of_users 		= len(user_number_of_orders)

## Number of Product Features. Include Features Directly from data
product_features 		= product_features
include_order_features 	= ['days_since_prior_order']
order_features 			= len(include_order_features)
total_features 			= product_features + order_features

## Print and Save Constants
print("Number of products: {}, Number of Users:{}, Time Steps:{}".format(number_of_products,number_of_users,time_steps))
total_data_shape = [number_of_users,time_steps,number_of_products,total_features]
user_shape = (1,time_steps,number_of_products,1)
user_val_shape = (1,1,number_of_products,1)
np.save('../../Processed_Data/input_dims',total_data_shape)

# for chunk in order_products_prior_df:
# 	# n += 1
# 	# print("Looping Through Chunk {}".format(n))
# 	# print(chunk)
# 	product_order_df = pd.merge(chunk,orders_df,on = 'order_id',how = 'left')
# 	break

def initialize_data():
	print("Creating Data Sets")
	open(data_file_path,'w').close()
	h5f = h5py.File(data_file_path,'a',libver = 'latest')
	h5f.create_dataset(train_set_name,user_shape,maxshape = (None,time_steps,number_of_products,1),chunks = user_shape,compression = "gzip",compression_opts = 9)
	h5f.create_dataset(val_set_name,(1,1,number_of_products,1),maxshape = (None,1,number_of_products,1),chunks = (1,1,number_of_products,1),compression = "gzip",compression_opts = 9)
	h5f.create_dataset(test_set_name,(1,1,number_of_products,1),maxshape = (None,1,number_of_products,1),chunks = (1,1,number_of_products,1),compression = "gzip",compression_opts = 9)
	h5f.create_dataset(val_user_set_name,data = np.zeros(number_of_users),compression = "gzip",compression_opts = 9)
	h5f.create_dataset(test_user_set_name,data = np.zeros(number_of_users),compression = "gzip",compression_opts = 9)

	print("Finished Creating Data Sets")

	print("Resizing")
	for dset in h5f.keys():
		# print(dset)
		h5f[dset].resize(number_of_users,axis = 0)
	print("Done Resizing")
	
	print("Initialize Values with 0")
	n = 0
	batch = 100
	start = time.time()
	while n < number_of_users:
		if(n + batch >= number_of_users):
			batch = number_of_users - n
		if(n % 500 == 0):
			
			print("{} users initialized. Time since last update: {}".format(n,(time.time() - start)))
			start = time.time()
		zero_time = time.time()
		init = np.zeros((batch,time_steps,number_of_products,1))
		print("{} Numpy Zeros Time".format(time.time()-zero_time))
		h5f_time = time.time()
		h5f[train_set_name][n:n+batch] = init
		h5f[val_set_name][n:n+batch] = init[:,0:1,:,:]
		h5f[test_set_name][n:n+batch] = init[:,0:1,:,:]
		print("{} Save Batch Time".format(time.time()-zero_time))
		n += batch
	print("Done Initializing")
	h5f.close()

def setup_DSPO():
	print("Add DSPO")

	init_data = np.zeros((number_of_users,time_steps))
	init_val_and_test = np.zeros((number_of_users,1))
	h5f = h5py.File(data_file_path,'a',libver = 'latest')
	h5f.create_dataset(train_DSPO_set,data = init_data,compression = "gzip",compression_opts = 9)
	h5f.create_dataset(val_DSPO_set,data = init_val_and_test,compression = "gzip",compression_opts = 9)
	h5f.create_dataset(test_DSPO_set,data = init_val_and_test,compression = "gzip",compression_opts = 9)

	start = time.time()
	n = 0
	for order in orders_df.itertuples():
		# print("{} DSPOs Added. Time since last update: {}".format(n,(time.time() - start)))
		# start = time.time()
		if(n % 10000 == 0):	
			print("{} DSPOs Added. Time since last update: {}".format(n,(time.time() - start)))
			start = time.time()
		n += 1
		user = int(order.user_id - 1)
		if(order.eval_set == train):
			time_step = time_steps - 1 - int(user_number_of_orders[order.user_id] - 1) + int(order.order_number - 1)
		else:
			time_step = 0
		if(order.order_number > 1):
			DSPO = np.array(order.days_since_prior_order)
			if(order.eval_set == train):
				h5f[train_DSPO_set][user,time_step] = DSPO
			if(order.eval_set == val):
				h5f[val_DSPO_set][user,time_step] = DSPO
				h5f[val_user_set_name][user] = np.array(1)
				# print("USer: {}, DSPO: {}".format(user,DSPO[0,0,0,0]))
			if(order.eval_set == test):
				h5f[test_DSPO_set][user,time_step] = DSPO
				h5f[test_user_set_name][user] = np.array(1)
	print("Done Adding DSPO. Time Since Last Update: {}".format(time.time()-start))

def setup_product_orders():
	h5f = h5py.File(data_file_path,'a',libver = 'latest')
	n = 0
	fill_value = np.array(1)
	start = time.time()
	for chunk in order_products_prior_df:
		n += 1
		print("Looping Through prior_df Chunk {}. Time since last chunk: {}".format(n,time.time() - start))
		start = time.time()
		# print(chunk.head())
		product_order_df = pd.merge(chunk,orders_df,on = 'order_id',how = 'left')
		product_order_df = product_order_df[product_order_df.user_id.notnull()]
		for product_order in product_order_df.itertuples():
			# print(product_order)
			user = int(product_order.user_id - 1)
			product = int(product_order.product_id - 1)
			if(product_order.eval_set == train):
				time_step = time_steps - 1 - int(user_number_of_orders[product_order.user_id] - 1) + int(product_order.order_number - 1)
			else:
				time_step = 0
			if(product_order.eval_set == train):
				h5f[train_set_name][user,time_step,product,0] = fill_value
			if(product_order.eval_set == val):
				h5f[val_set_name][user,time_step,product,0] = fill_value
		if(Tst):
			break
	n = 0
	for chunk in order_products_train_df:
		n += 1
		print("Looping Through train_df Chunk {}. Time since last chunk: {}".format(n,time.time() - start))
		start = time.time()
		product_order_df = pd.merge(chunk,orders_df,on = 'order_id',how = 'left')
		product_order_df = product_order_df[product_order_df.user_id.notnull()]
		for product_order in product_order_df.itertuples():
			user = int(product_order.user_id - 1)
			product = int(product_order.product_id - 1)
			if(product_order.eval_set == train):
				time_step = time_steps - 1 - int(user_number_of_orders[product_order.user_id] - 1) + int(product_order.order_number - 1)
			else:
				time_step = 0
			if(product_order.eval_set == train):
				h5f[train_set_name][user,time_step,product,0] = fill_value
			if(product_order.eval_set == val):
				# print("valing: {}".format(product_order.user_id))
				h5f[val_set_name][user,time_step,product,0] = fill_value
		if(Tst):
			break
	h5f.close()


## Test Data File
def testing_file():

	input_df = pd.merge(orders_df,order_products_df_test,on = 'order_id',how = 'left')	
	input_df = input_df[input_df.product_id.notnull()]
	h5f = h5py.File(data_file_path,'a')


	# print(h5f[train_set_name].shape)
	for _,order in input_df.iterrows():
		# print(order)
		# print(order.order_number)
		time_step = int(time_steps - 1 - int(user_number_of_orders[order.user_id] - 1) + int(order.order_number - 1))
		if(order.eval_set == 'train'):
			if(h5f[val_set_name][int(order.user_id - 1),0,int(order.product_id - 1),0] < 1):
				print("Val Product Error at: {}".format((int(order.user_id),0,int(order.product_id - 1),0)))
			if(np.sum(np.abs(h5f[val_DSPO_set][int(order.user_id - 1),0] - order.days_since_prior_order)) > 0):
				# print(order.user_id)
				print("Val DSPO Error at: {}, Diff = {}".format(int(order.user_id),np.sum(h5f[val_DSPO_set][int(order.user_id - 1),0] - order.days_since_prior_order)))
		if(order.eval_set == 'prior'):
			# print(order.order_number)
			# print(np.sum(np.abs(h5f[train_set_name][int(order.user_id - 1),time_step,:,1])))
			if(h5f[train_set_name][int(order.user_id - 1),time_step,int(order.product_id - 1),0] < 1):
				print("Train Product Error at: {}".format((int(order.user_id),time_step,int(order.product_id - 1),0)))
			if(np.sum(np.abs(h5f[train_DSPO_set][int(order.user_id - 1),time_step] - order.days_since_prior_order)) > 0):
				print("Train DSPO Error at: {}, Diff = {}".format((int(order.user_id),order.order_number),np.sum(h5f[train_DSPO_set][int(order.user_id - 1),time_step] - order.days_since_prior_order)))

	test_df = orders_df[(orders_df.eval_set == 'test')]	
	for _,order in test_df.iterrows():
		if(order.eval_set == 'test'):
			if(np.sum(np.abs(h5f[test_DSPO_set][int(order.user_id - 1),0] - order.days_since_prior_order)) > 0):
				print("Test DSPO Error at: {}, Diff = {}".format(int(order.user_id),np.sum(h5f[test_DSPO_set][int(order.user_id - 1),0] - order.days_since_prior_order)))

	h5f.close()

def test_user_list():
	h5f = h5py.File(data_file_path,'a')
	test_users = h5f[test_user_set_name]
	test_set = h5f[test_DSPO_set]
	val_users = h5f[val_user_set_name]
	val_set = h5f[val_DSPO_set]

	for n in range(test_set.shape[0]):
		test_sum = np.sum(test_set[n,:])
		test_user_fail = ( test_sum > 0 and int(test_users[n]) == 0 ) or (int(test_sum) == 0  and test_users[n] > 0)
		val_sum = np.sum(val_set[n,:])
		val_user_fail = ( val_sum > 0 and int(val_users[n]) == 0 ) or (int(val_sum) == 0  and val_users[n] > 0)
		if(val_user_fail or test_user_fail):
		# if(True):	
			print("n: {}, Val: {}, Test: {}, Val Sum: {}, Val User: {}".format(n,val_user_fail,test_user_fail,val_sum,val_users[n]))



def quick_test():
	h5f = h5py.File(data_file_path,'a')
	print("Train Shape: {}, Val Shape: {}, Test Shape: {}".format(h5f[train_set_name].shape,h5f[val_set_name].shape,h5f[test_set_name].shape))
	# print(np.shape(h5f[train_set_name][54:55]))

initialize_data()
setup_DSPO()
setup_product_orders()

print("Testing")
quick_test()
testing_file()
test_user_list()
