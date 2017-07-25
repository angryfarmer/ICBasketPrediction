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


# order_products_train_df = pd.merge(order_products_train_df,products_df, on = 'product_id', how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,orders_df, on = 'order_id', how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,aisles_df,on = 'aisle_id',how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,departments_df,on = 'department_id',how = 'left')

# order_products_prior_df = pd.merge(order_products_prior_df,products_df, on = 'product_id', how = 'left')
# order_products_prior_df = pd.merge(order_products_prior_df,orders_df, on = 'order_id', how = 'left')
# order_products_prior_df = pd.merge(order_products_prior_df,aisles_df,on = 'aisle_id',how = 'left')
# order_products_prior_df = pd.merge(order_products_prior_df,departments_df,on = 'department_id',how = 'left')

# order_products_prior_df.append(order_products_train_df)
del order_products_train_df

def get_unique_users():
	return orders_df.loc[:,'user_id'].unique()

def get_previous_orders(order_id,user_df):

	# Get Rows for Previous Orders based on smaller order_number
	selected_order = user_df[user_df['order_id'] == order_id]
	order_number = selected_order.order_number.iloc[0]
	user_id = selected_order.user_id.iloc[0]
	previous_orders = user_df[(user_df['user_id'] == user_id) & (user_df['order_number'] <= order_number)]
	previous_orders = previous_orders.order_id.tolist()
	
	return(previous_orders)

def days_since_prior_item_order(product_ids,prev_orders,user_df,user_df_with_prior):

	# Add current time difference at the end
	latest_order = prev_orders.pop()
	days_since_last_order_latest = user_df[user_df['order_id'] == latest_order].days_since_prior_order.iloc[0]

	# Merge and Query Dataframes for relevant information
	df = user_df[user_df['order_id'].isin(prev_orders)]
	df_with_prior = user_df_with_prior[user_df_with_prior['order_id'].isin(prev_orders)]
	# df_with_prior = df_with_prior.merge(df,on='order_id',how = 'left')
	
	# Calculate and create list
	days_since_prior_item_order_list = []
	for product_id in product_ids:
		last_order_number = (df_with_prior[df_with_prior['product_id'] == product_id])['order_number'].max()
		days_since_order = df[df['order_number'] > last_order_number]['days_since_prior_order'].sum()
		days_since_order = days_since_order + days_since_last_order_latest
		days_since_prior_item_order_list.append(days_since_order)
	
	# Restore originial order list
	prev_orders.append(latest_order)

	return(days_since_prior_item_order_list)


def create_test_case_for_order(queried_order,user_df,user_df_with_prior):
	
	prev_orders = get_previous_orders(queried_order,user_df)
	# print(prev_orders)
	# Remove the current order being examined from the list
	test_order = prev_orders.pop()
	test_order_days_since_last_order = user_df[user_df['order_id'] == test_order].days_since_prior_order.iloc[0]

	# Find all items ordered previously
	df = user_df[user_df['order_id'].isin(prev_orders)]
	df_with_prior = user_df_with_prior[user_df_with_prior['order_id'].isin(prev_orders)]
	# df_with_prior = df_with_prior.merge(df,on='order_id',how = 'left')
	
	# Create new data frame with User, Item, Order ID, Days_since_last _order and label
	test_order_products = user_df_with_prior[user_df_with_prior['order_id'] == test_order].product_id.unique()
	y_label = df_with_prior.product_id.drop_duplicates().isin(test_order_products)
	new_test_df = df_with_prior.drop_duplicates('product_id')[['user_id','product_id','order_id','days_since_prior_order']]
	new_test_df['label'] = y_label
	new_test_df['order_id'] = test_order

	# Generate Previous Order Days Orders
	prev_orders.append(test_order)
	previous_order_days = days_since_prior_item_order(new_test_df.product_id,prev_orders,user_df,user_df_with_prior)
	new_test_df['days_since_prior_order'] = previous_order_days

	return(new_test_df)

# sets = orders_df.eval_set.unique()
# print(sets)


n = 0
training_df = orders_df[orders_df['eval_set'].isin(["train","prior"])]
# print(len(training_df))

with open('input_file.csv','a') as f:
	user_id = 0
	for index,row in training_df.iterrows():
		# print(row)
		# print(row.user_id)
		if(not row.user_id == user_id):
			user_id = row.user_id
			user_df = orders_df[orders_df.user_id == user_id]
			user_df_with_prior = order_products_prior_df[order_products_prior_df['order_id'].isin(user_df.order_id.unique())]
			user_df_with_prior = user_df_with_prior.merge(user_df,on='order_id',how='left')
			n += 1
		order_id = row.order_id
		
		write_df = create_test_case_for_order(order_id,user_df,user_df_with_prior)
		# print(write_df)
		if n % 500 == 0:
			print(n)
			print(index)
		write_df.to_csv(f, header=False)



# create_test_case_for_order(2539329)

# print(orders_df.head())
# cnt_srs = orders_df.eval_set.value_counts()

# plt.figure(figsize = (12,8))
# sns.barplot(cnt_srs.index, cnt_srs.values,alpha = 0.8)
# plt.ylabel("Number of Occurances",fontsize = 12)
# plt.xlabel("Eval Set Type", fontsize = 12)
# plt.xticks(rotation = "vertical")
# plt.show()


def get_unique_count(x):
	return len(np.unique(x))

# cnt_srs = orders_df.groupby("eval_set")
# # print(cnt_srs.get_group('test').head())
# print(cnt_srs['user_id'].aggregate(get_unique_count))

# plt.figure(figsize = (12,8))
# sns.countplot(x="eval_set",data = orders_df)
# # plt.show()

# order_products_train_df = pd.merge(order_products_train_df,products_df, on = 'product_id', how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,orders_df, on = 'order_id', how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,aisles_df,on = 'aisle_id',how = 'left')
# order_products_train_df = pd.merge(order_products_train_df,departments_df,on = 'department_id',how = 'left')
# # temp_df = order_products_train_df[['order_id','user_id']]
# # print(order_products_train_df.head(20))
# # train_ids = order_products_train_df.loc[:,'user_id'].unique()
# # print(order_products_prior_df.head(20))
# # print(order_products_train_df.loc[:5,['order_id','user_id','days_since_prior_order']].as_matrix())

# # print(train_ids[:30])

# order_products_prior_df = pd.merge(order_products_prior_df,products_df, on = 'product_id', how = 'left')
# order_products_prior_df = pd.merge(order_products_prior_df,orders_df, on = 'order_id', how = 'left')
# # order_products_prior_df = order_products_prior_df[order_products_prior_df['user_id'].isin(train_ids)]
# order_products_prior_df = pd.merge(order_products_prior_df,aisles_df,on = 'aisle_id',how = 'left')
# order_products_prior_df = pd.merge(order_products_prior_df,departments_df,on = 'department_id',how = 'left')
# # temp_ids = (order_products_prior_df.loc[:,'user_id'].unique())

# print(len(temp_ids))
# a = 0
# for n in temp_ids:
# 	if(not n in train_ids):
# 		a += 1
# print(a)
