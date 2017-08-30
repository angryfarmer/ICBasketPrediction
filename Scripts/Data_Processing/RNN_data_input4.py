import numpy as np
import pandas as pd
import os
import h5py

Test_Set = False
product_features = 1
inpath_str = "..\\..\\Data\\"
data_file_path = "..\\..\\Processed_Data\\user_baskets_loc"
val_set_path = "..\\..\\Processed_Data\\val_set"
test_set_path = "..\\..\\Processed_Data\\test_set"
train_set_name = "train_set"
val_set_name = "val_set"
test_set_name = "test_set"
val_user_set_name = "val_users"
test_user_set_name = "test_users"
set_information = "set_info"

order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
products_df             = pd.read_csv(inpath_str + 'products.csv')
orders_df               = pd.read_csv(inpath_str + 'orders.csv')

## Orders DF setup
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

print(orders_df.eval_set.value_counts())

def initialize_file():
	open(data_file_path,'w').close()
	h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(set_information,data = np.array([number_of_users,time_steps,number_of_products]))

def train_set_data():
	h5f = h5py.File(data_file_path,'a')

	train_df = pd.merge(order_products_prior_df,orders_df,on = 'order_id', how = 'left')[['user_id','order_number','product_id','days_since_prior_order']]
	train_df = train_df.sort_values('user_id')
	print(train_df.user_id.is_monotonic)
	h5f.create_dataset(train_set_name,data = train_df.as_matrix().astype(np.int64))
	print(h5f[train_set_name][:3])
	print(h5f[train_set_name].shape)


def val_set_data():
	h5f = h5py.File(data_file_path,'a')
	val_df = pd.merge(order_products_train_df,orders_df,on = 'order_id', how = 'left')[['user_id','order_number','product_id','days_since_prior_order']]
	val_df = val_df.sort_values('user_id')
	print(val_df.user_id.is_monotonic)
	h5f.create_dataset(val_set_name,data = val_df.as_matrix().astype(np.int64))
	print(h5f[val_set_name][:3])
	print(h5f[val_set_name].shape)

def test_set_input():
	h5f = h5py.File(data_file_path,'a')
	test_df = orders_df[orders_df.eval_set == 'test'][['user_id','order_number','days_since_prior_order']]
	test_df = test_df.sort_values('user_id')
	h5f.create_dataset(test_set_name,data = test_df.as_matrix().astype(np.int64))
	print(h5f[test_set_name][:3])
	print(h5f[test_set_name].shape)


initialize_file()
train_set_data()
val_set_data()
test_set_input()