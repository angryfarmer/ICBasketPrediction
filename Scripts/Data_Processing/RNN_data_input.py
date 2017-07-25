import numpy as np
import pandas as pd
# import scipy as sci
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf

base_folder_path = "C:\\Users\\Eric\\Documents\\Projects\\Tensorflow\\ICBasketPrediction"
data_folder_path = "\\Data\\"
inpath_str 	 = base_folder_path + data_folder_path

aisles_df               = pd.read_csv(inpath_str + 'aisles.csv')
departments_df          = pd.read_csv(inpath_str + 'departments.csv')
order_products_prior_df = pd.read_csv(inpath_str + 'order_products__prior.csv',chunksize = 500000)
order_products_prior_df = pd.concat(order_products_prior_df,ignore_index = True)
order_products_train_df = pd.read_csv(inpath_str + 'order_products__train.csv',chunksize = 500000)
order_products_train_df = pd.concat(order_products_train_df, ignore_index = True)
order_products_prior_df = pd.concat([order_products_prior_df,order_products_train_df],ignore_index = True)
orders_df               = pd.read_csv(inpath_str + 'orders.csv')
products_df             = pd.read_csv(inpath_str + 'products.csv')


##### Orders DF setup #####
 
# print(orders_df.head())
# user_orders = orders_df.groupby(['user_id']).size()
# print(user_orders.head())
# print(user_orders.head().max())
# print(len(user_orders))
# orders_details_df = pd.merge(orders_df,order_products_prior_df,on = 'order_id',how = 'left')
# print(orders_details_df.head(20))


##### Products DF Setup ######

print(products_df.head())
print(len(products_df))
print(products_df.product_id.max())

# basket represented by a k by product_rank matrix where:
#    k = number of unique products
#    product_rank = number of features each product can have


