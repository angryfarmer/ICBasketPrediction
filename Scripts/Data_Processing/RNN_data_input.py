import numpy as np
import pandas as pd
import os
import h5py

Test_Set = True
product_features = 1
inpath_str = "..\\..\\Data\\"
data_file_path = "..\\..\\Processed_Data\\user_baskets"
val_set_path = "..\\..\\Processed_Data\\val_set"
test_set_path = "..\\..\\Processed_Data\\test_set"
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
val_user_set_name = "val_users"
test_user_set_name = "test_users"

if(not Test_Set):
	##### Training Data #####
	aisles_df               = pd.read_csv(inpath_str + 'aisles.csv')
	departments_df          = pd.read_csv(inpath_str + 'departments.csv')
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
	order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
	order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
	order_products_df 		= pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv')
	products_df             = pd.read_csv(inpath_str + 'products.csv')
else:
	##### Test Data #####
	aisles_df               = pd.read_csv(inpath_str + 'aisles.csv',nrows = 1000)
	departments_df          = pd.read_csv(inpath_str + 'departments.csv',nrows = 1000)
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',nrows = 500000)
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',nrows = 500000)
	order_products_df 		= pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv')
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
user_shape = (1,time_steps,number_of_products,total_features)
np.save('../../Processed_Data/input_dims',total_data_shape)



## Function to Initialize Product If In Basket.
def product_in_basket(order,user_basket,single_item_order = False):
	## Initialize product with ones
	product_initalizer = np.ones(total_features)
	## Overwrite one index for every feature to be included from orignal data
	if(single_item_order):
		time_step = 0
	else:
		time_step = time_steps - 1 - int(user_number_of_orders[order.user_id] - 1) + int(order.order_number - 1)
	
	for index,feature in enumerate(include_order_features):
		product_initalizer[index+product_features] = order[feature]
		if(user_basket[0,time_step,0,index + product_features] > 0 and index + product_features == 1):
			pass
		else:
			user_basket[0,time_step,:,index + product_features] = order[feature]
	## Add product to user at timestep (order.order_number-1)
	user_basket[0,time_step,int(order.product_id - 1),:] = product_initalizer

## Function to add days since last order
def add_days_since_last_order(order,user_basket,single_item_order = False):
	if(single_item_order):
		time_step = 0
	else:
		time_step = time_steps - 1 - int(user_number_of_orders[order.user_id] - 1) + int(order.order_number - 1)
	
	for index,feature in enumerate(include_order_features):
		user_basket[0,time_step,:,index + product_features] = order[feature]

## Function To Save User Baskets to File
def save_user_to_file(user,user_basket,dataset_name):
	data_shape = (user,user_shape[1],user_shape[2],user_shape[3])
	h5f = h5py.File(data_file_path,'a')
	h5f[dataset_name].resize(user,axis = 0)
	h5f[dataset_name][user-1:user,:,:,:] = user_basket

	sum_user = np.sum(h5f[dataset_name][user-1:user,:,:,:])
	sum_days = np.sum(h5f[dataset_name][user-1:user,:,0,1])
	sum_total_days = np.sum(h5f[dataset_name][user-1:user,:,:,1:])

	# print("User: {}, Sum Check {}, Features: {}, {}, {}".format(user,sum_user,sum_total_days/sum_days/number_of_products,sum_days,sum_total_days))
	# print(h5f[dataset_name][user-1:user,:,0,1])
	h5f.close()

## Run Through Data Frame To Generate Training File
def generate_train_set():
	## Initialize Data File
	train_df = pd.merge(orders_df[(orders_df.eval_set == 'prior')],order_products_df,on = 'order_id',how = 'left')	
	train_df = train_df[train_df.product_id.notnull()]
	open(data_file_path,'w').close()
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(train_set_name,user_shape,maxshape = (None,time_steps,number_of_products,total_features),chunks = user_shape,compression = "gzip",compression_opts = 9)
	h5f.close()

	## Initialize Basket
	user_basket = np.zeros(user_shape)

	## Run through data frame
	user = 1
	for index,item_order in train_df.iterrows():
		if(not int(item_order.user_id) == user):
			## Save User And Reset For Next User. Works due to monotonicity of user_id in dataframe
			save_user_to_file(user,user_basket,train_set_name)
			user_basket = np.zeros(user_shape)
			
			## Used for Filling in Gaps Due to Test Set
			for n in range(int(item_order.user_id) - user - 1):
				save_user_to_file(user + n + 1,user_basket,train_set_name)
				user_basket = np.zeros(user_shape)
			
			## Move to next user index
			user = int(item_order.user_id)

		## Condition exists due to test set's sparse data
		if(np.isnan(item_order.product_id) or item_order.product_id > number_of_products):
			pass
		else:
			product_in_basket(item_order,user_basket)
	
	## Save last user at end of dataframe
	save_user_to_file(user,user_basket,train_set_name)
	user_basket = np.zeros(user_shape)

	## Append user data due to sparse test set
	for n in range(int(number_of_users) - user - 1):
		save_user_to_file(user + n + 1,user_basket,train_set_name)
		user_basket = np.zeros(user_shape)

## Run Through Data Frame To Generate Val File
def generate_val_set():
	## Initialize Data File
	user_shape = (1,1,number_of_products,total_features)
	val_df = pd.merge(orders_df[(orders_df.eval_set == 'train')],order_products_df,on = 'order_id',how = 'left')	
	val_df = val_df[val_df.product_id.notnull()]
	h5f = h5py.File(data_file_path,'a')
	h5f.__delitem__(val_set_name)
	h5f.__delitem__(val_user_set_name)
	h5f.create_dataset(val_set_name,user_shape,maxshape = (None,1,number_of_products,total_features),chunks = user_shape,compression = "gzip",compression_opts = 9)
	h5f.close()

	## Initialize Array to capture users in the val set
	val_users = np.zeros(number_of_users)

	## Initialize Basket
	user_basket = np.zeros(user_shape)

	## Run through data frame
	user = 1
	for index,item_order in val_df.iterrows():
		## Add user from order to the val set
		val_users[int(item_order.user_id - 1)] = 1
		print(int(item_order.user_id))
		# print(val_users[int(item_order.user_id-1)])

		##Action when user changes
		if(not int(item_order.user_id) == user):
			## Save User And Reset For Next User. Works due to monotonicity of user_id in dataframe
			save_user_to_file(user,user_basket,val_set_name)
			user_basket = np.zeros(user_shape)
			
			## Used for Filling in Gaps if user does not make an order in this set
			for n in range(int(item_order.user_id) - user - 1):
				save_user_to_file(user + n + 1,user_basket,val_set_name)
				user_basket = np.zeros(user_shape)
			
			## Move to next user index
			user = int(item_order.user_id)

		## Condition exists due to script testing set data
		if(np.isnan(item_order.product_id) or item_order.product_id > number_of_products):
			pass
		else:
			product_in_basket(item_order,user_basket,single_item_order = True)
	
	## Save last user at end of dataframe
	save_user_to_file(user,user_basket,val_set_name)
	user_basket = np.zeros(user_shape)

	## Save val users
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(val_user_set_name,data = val_users)
	h5f.close()
	## Append user data due to sparse test set
	for n in range(int(number_of_users) - user - 1):
		save_user_to_file(user + n + 1,user_basket,val_set_name)
		user_basket = np.zeros(user_shape)

def generate_test_set():
	## Initialize Data File
	user_shape = (1,1,number_of_products,total_features)
	test_df = orders_df[(orders_df.eval_set == 'test')]	
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(test_set_name,user_shape,maxshape = (None,1,number_of_products,total_features),chunks = user_shape,compression = "gzip",compression_opts = 9)
	h5f.close()

	## Initialize Array to capture users in the test set
	test_users = np.zeros(number_of_users)

	## Initialize Basket
	user_basket = np.zeros(user_shape)

	## Run through data frame
	user = 1
	for index,item_order in test_df.iterrows():
		## Add order user to test set
		test_users[int(item_order.user_id - 1)] = 1

		if(not int(item_order.user_id) == user):
			## Save User And Reset For Next User. Works due to monotonicity of user_id in dataframe
			save_user_to_file(user,user_basket,test_set_name)
			user_basket = np.zeros(user_shape)
			
			## Used for Filling in Gaps Due to Test Set
			for n in range(int(item_order.user_id) - user - 1):
				save_user_to_file(user + n + 1,user_basket,test_set_name)
				user_basket = np.zeros(user_shape)
			
			## Move to next user index
			user = int(item_order.user_id)

		add_days_since_last_order(item_order,user_basket,single_item_order = True)
	
	## Save last user at end of dataframe
	save_user_to_file(user,user_basket,test_set_name)
	user_basket = np.zeros(user_shape)

	## Save val users
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(test_user_set_name,data = test_users)
	h5f.close()

	## Append user data due to sparse test set
	for n in range(int(number_of_users) - user - 1):
		save_user_to_file(user + n + 1,user_basket,test_set_name)
		user_basket = np.zeros(user_shape)


## Test Data File
def testing_file():
	input_df = pd.merge(orders_df,order_products_df,on = 'order_id',how = 'left')	
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
			if(np.sum(np.abs(h5f[val_set_name][int(order.user_id - 1),0,:,1] - order.days_since_prior_order)) > 0):
				# print(order.user_id)
				print("Val DSPO Error at: {}, Diff = {}".format(int(order.user_id),np.sum(h5f[val_set_name][int(order.user_id - 1),0,:,1] - order.days_since_prior_order)))
		if(order.eval_set == 'prior'):
			if(h5f[train_set_name][int(order.user_id - 1),time_step,int(order.product_id - 1),0] < 1):
				print("Train Product Error at: {}".format((int(order.user_id),time_step,int(order.product_id - 1),0)))
			if(np.sum(np.abs(h5f[train_set_name][int(order.user_id - 1),time_step,:,1] - order.days_since_prior_order)) > 0):
				print("Train DSPO Error at: {}, Diff = {}".format((int(order.user_id),order.order_number),np.sum(h5f[train_set_name][int(order.user_id - 1),time_step,:,1] - order.days_since_prior_order)))

	test_df = orders_df[(orders_df.eval_set == 'test')]	
	for _,order in test_df.iterrows():
		if(order.eval_set == 'test'):
			if(np.sum(np.abs(h5f[test_set_name][int(order.user_id - 1),0,:,1] - order.days_since_prior_order)) > 0):
				print("Test DSPO Error at: {}, Diff = {}".format(int(order.user_id),np.sum(h5f[test_set_name][int(order.user_id - 1),time_step,:,1] - order.days_since_prior_order)))

	h5f.close()

def test_user_list():
	h5f = h5py.File(data_file_path,'a')
	test_users = h5f[test_user_set_name]
	test_set = h5f[test_set_name]
	val_users = h5f[val_user_set_name]
	val_set = h5f[val_set_name]

	for n in range(test_set.shape[0]):
		test_sum = np.sum(test_set[n,:,:,1])
		test_user_fail = ( test_sum > 0 and int(test_users[n]) == 0 ) or (int(test_sum) == 0  and test_users[n] > 0)
		val_sum = np.sum(val_set[n,:,:,0])
		val_user_fail = ( val_sum > 0 and int(val_users[n]) == 0 ) or (int(val_sum) == 0  and val_users[n] > 0)
		if(val_user_fail or test_user_fail):

			print("n: {}, Val: {}, Test: {}".format(n,val_user_fail,test_user_fail))



def quick_test():
	h5f = h5py.File(data_file_path,'a')
	print("Train Shape: {}, Val Shape: {}, Test Shape: {}".format(h5f[train_set_name].shape,h5f[val_set_name].shape,h5f[test_set_name].shape))
	# print(np.shape(h5f[train_set_name][54:55]))

# generate_train_set()
# print("++++++++")
# generate_val_set()
# print("++++++++")
# generate_test_set()

# quick_test()
# testing_file()
test_user_list()
