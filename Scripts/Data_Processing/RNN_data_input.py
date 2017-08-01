import numpy as np
import pandas as pd
import os
# import scipy as sci
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf

inpath_str = "..\\..\\Data\\"

##### Actual Data #####
# aisles_df               = pd.read_csv(inpath_str + 'aisles.csv')
# departments_df          = pd.read_csv(inpath_str + 'departments.csv')
# order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
# order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
# order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
# order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
# order_products_prior_df = pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
# orders_df               = pd.read_csv(inpath_str + 'orders.csv')
# products_df             = pd.read_csv(inpath_str + 'products.csv')

##### Test Data #####
aisles_df               = pd.read_csv(inpath_str + 'aisles.csv',nrows = 1000)
departments_df          = pd.read_csv(inpath_str + 'departments.csv',nrows = 1000)
order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',nrows = 20000)
order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',nrows = 20000)
order_products_prior_df = pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
orders_df               = pd.read_csv(inpath_str + 'orders.csv',nrows = 1000)
products_df             = pd.read_csv(inpath_str + 'products.csv')

##### Orders DF setup #####
# print(orders_df.head())
user_number_of_orders = orders_df.groupby(['user_id']).size()


orders_details_df = pd.merge(orders_df,order_products_prior_df,on = 'order_id',how = 'left')
time_steps = user_number_of_orders.head().max()
number_of_products = len(products_df) 
number_of_users = len(user_number_of_orders)
product_features = 10
include_order_features = ['days_since_prior_order']
order_features = len(include_order_features)
total_features = product_features + order_features

print("Number of products: {}, Number of Users:{}, Time Steps:{}".format(number_of_products,number_of_users,time_steps))
# print(orders_df.user_id.max())
# print(number_of_users)

# print(orders_details_df.head(20))


##### Products DF Setup ######
# basket represented by a k by product_rank matrix where:
#    k = number of unique products
#    product_rank = number of features each product can have

# print(products_df.head())
# print(number_of_products)
# print(products_df.product_id.max())

input_data = np.zeros((number_of_users,time_steps,number_of_products,total_features))
dims = [number_of_users,time_steps,number_of_products,total_features]
np.save('../../Processed_Data/input_dims',dims)
def product_in_basket(user,time_step,product,order):
	product_initalizer = np.ones(total_features)
	for index,feature in enumerate(include_order_features):
		product_initalizer[product_features + index] = order[feature]
	input_data[int(user),int(time_step),int(product),:] = product_initalizer

# print(order_products_prior_df.head(5))
for index,row in orders_details_df.iterrows():
	if(np.isnan(row.product_id) or row.product_id > number_of_products):
		pass
	else:
		product_in_basket(row.user_id - 1,row.order_number - 1,row.product_id - 1,row)

# print((input_data[54,7,7735,:]))

np.save('../../Processed_Data/instacart_data_input',input_data)
# in_data = np.load('instacart_data_input.npy')
# print(in_data[54,7,7735,:])