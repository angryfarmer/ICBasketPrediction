import numpy as np
import pandas as pd
import os
import tables
import h5py

Test_Set = True
product_features = 10
inpath_str = "..\\..\\Data\\"
data_file_path = "..\\..\\Processed_Data\\user_baskets"
dataset_name = "rnn_data_input"

if(not Test_Set):
	##### Training Data #####
	aisles_df               = pd.read_csv(inpath_str + 'aisles.csv')
	departments_df          = pd.read_csv(inpath_str + 'departments.csv')
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
	order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
	order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
	order_products_prior_df = pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv')
	products_df             = pd.read_csv(inpath_str + 'products.csv')
else:
	##### Test Data #####
	aisles_df               = pd.read_csv(inpath_str + 'aisles.csv',nrows = 1000)
	departments_df          = pd.read_csv(inpath_str + 'departments.csv',nrows = 1000)
	order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',nrows = 20000)
	order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',nrows = 20000)
	order_products_prior_df = pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
	orders_df               = pd.read_csv(inpath_str + 'orders.csv',nrows = 1000)
	products_df             = pd.read_csv(inpath_str + 'products.csv')

## Orders Details DF setup
orders_df 				= orders_df.sort_values('user_id')
orders_details_df 		= pd.merge(orders_df,order_products_prior_df,on = 'order_id',how = 'left')
print("Is Monotonic: {}".format(orders_details_df.user_id.is_monotonic))
orders_details_df = orders_details_df[orders_details_df.product_id.notnull()]

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
def product_in_basket(order,user_basket):
	## Initialize product with ones
	product_initalizer = np.ones(total_features)
	## Overwrite one index for every feature to be included from orignal data
	for index,feature in enumerate(include_order_features):
		product_initalizer[product_features + index] = order[feature]
	## Add product to user at timestep (order.order_number-1)
	user_basket[0,int(order.order_number - 1),int(order.product_id - 1),:] = product_initalizer

## Function To Save User Baskets to File
def save_user_to_file(user,user_basket):
	data_shape = (user,user_shape[1],user_shape[2],user_shape[3])
	h5f = h5py.File(data_file_path,'a')
	h5f[dataset_name].resize(user,axis = 0)
	h5f[dataset_name][user-1:user,:,:,:] = user_basket
	h5f.close()


## Run Through Data Frame To Generate File
def generate_file():
	## Initialize Data File
	open(data_file_path,'w').close()
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(dataset_name,user_shape,maxshape = (None,time_steps,number_of_products,total_features),chunks = user_shape,compression = "gzip",compression_opts = 9)
	h5f.close()

	## Initialize Basket
	user_basket = np.zeros(user_shape)

	## Run through data frame
	user = 1
	for index,row in orders_details_df.iterrows():
		if(not int(row.user_id) == user):
			## Save User And Reset For Next User. Works due to monotonicity of user_id in dataframe
			save_user_to_file(user,user_basket)
			user_basket = np.zeros(user_shape)
			
			## Used for Filling in Gaps Due to Test Set
			for n in range(int(row.user_id) - user - 1):
				save_user_to_file(user + n + 1,user_basket)
				user_basket = np.zeros(user_shape)
			
			## Move to next user index
			user = int(row.user_id)

		## Condition exists due to test set's sparse data
		if(np.isnan(row.product_id) or row.product_id > number_of_products):
			pass
		else:
			product_in_basket(row,user_basket)
	
	## Save last user at end of dataframe
	save_user_to_file(user,user_basket)
	user_basket = np.zeros(user_shape)

## Test Data File
def test_file():
	h5f = h5py.File(data_file_path,'a')
	print(h5f[dataset_name].shape)
	for user in range(56):
		print(np.sum(h5f[dataset_name][user:user+1,:,:,:]))
	h5f.close()

generate_file()
test_file()