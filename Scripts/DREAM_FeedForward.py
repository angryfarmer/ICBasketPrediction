## Need to create way to measure f score
## 	

import tensorflow as tf
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("data_importer", "Data_Processing\\data_importer.py")
data_importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_importer)


## Load input basket data
data_loader = data_importer.data_importer('..\\Processed_Data\\user_baskets',load_batch = 30)

## Input dimension sizes
in_dims						= np.load('..\\Processed_Data\\input_dims.npy')
number_of_products 			= in_dims[2]
number_of_data_features 	= in_dims[3]
number_of_users 			= in_dims[0]

# print(in_dims)
## Ranks for Each Layer
number_of_user_based_features 	= 1
number_of_product_features 		= 10
number_of_input_features 		= number_of_product_features + number_of_user_based_features
L_1_nodes 						= number_of_input_features

## Weight Initialization for LSTM
W_1_i 	= tf.Variable(tf.truncated_normal([number_of_input_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_input")
W_1_c 	= tf.Variable(tf.truncated_normal([number_of_input_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_candidate")
W_1_f 	= tf.Variable(tf.truncated_normal([number_of_input_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_forget")
W_1_o 	= tf.Variable(tf.truncated_normal([number_of_input_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_output")
W_h_i 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_input")
W_h_c 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_candidate")
W_h_f 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_forget")
W_h_o 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_output")
W_2 	= tf.Variable(tf.truncated_normal([L_1_nodes,number_of_product_features],stddev = 0.5, mean = 0),name = "W_2_User_Preference")

## Product Feature Initialization
product_weights = tf.Variable(tf.truncated_normal([number_of_products,number_of_product_features],stddev = 0.5,mean = 0),name = "product_weights")

def feed_foward_user(number_of_time_steps):

	## Placeholders to run graph
	input_data						= tf.placeholder(tf.float32,[None,number_of_time_steps,number_of_products,number_of_data_features])
	prev_cell_state 				= tf.zeros([tf.shape(input_data)[0],L_1_nodes])
	prev_hidden_layer 				= tf.zeros([tf.shape(input_data)[0],L_1_nodes])
	cost 							= tf.reduce_sum(tf.zeros([1]))
	prev_user_basket_representation = tf.zeros([tf.shape(input_data)[0],number_of_product_features])

	for n in range(number_of_time_steps - 1):
		with tf.name_scope("Time_Step_{}".format(n)):
			## Prepare input using max of each feature over last n baskets
			with tf.name_scope("Basket_Representation_{}".format(n)):
				user_baskets_0_to_n 			= tf.slice(input_data,[0,n,0,0],[-1,1,-1,1])
				user_baskets_0_to_n 			= tf.concat([user_baskets_0_to_n] * number_of_product_features,3)
				user_baskets_0_to_n 			= tf.multiply(user_baskets_0_to_n,product_weights)
				user_baskets_0_to_n 			= tf.reduce_max(user_baskets_0_to_n,2)
				user_baskets_0_to_n				= tf.concat([user_baskets_0_to_n,tf.expand_dims(prev_user_basket_representation,1)],1)
				user_basket_representation 		= tf.reduce_max(user_baskets_0_to_n,1)
				user_factors 					= tf.squeeze(tf.slice(input_data,[0,n+1,0,1],[-1,1,1,1]),[1,2])
				user_input 						= tf.concat([user_basket_representation,user_factors],1)
				prev_user_basket_representation = tf.identity(user_basket_representation)
			
			## Basic LSTM Cell setup
			with tf.name_scope("Cell_Gates_{}".format(n)):
				forget_gate 				= tf.sigmoid(tf.add(tf.matmul(user_input,W_1_f),tf.matmul(prev_hidden_layer,W_h_f)))
				input_gate 					= tf.sigmoid(tf.add(tf.matmul(user_input,W_1_i),tf.matmul(prev_hidden_layer,W_h_i)))
				candidate_value 			= tf.tanh(tf.add(tf.matmul(user_input,W_1_c),tf.matmul(prev_hidden_layer,W_h_c)))
				new_cell_state 				= tf.add(tf.multiply(input_gate,candidate_value),tf.multiply(forget_gate,prev_cell_state))
			
			## Generated User Preference
			with tf.name_scope("Cell_Outputs_{}".format(n)):
				L_1_output 					= tf.sigmoid(tf.add(tf.matmul(user_input,W_1_o),tf.matmul(prev_hidden_layer,W_h_o)))

			## Save Cell States and Hidden Layer for next Time Step
			with tf.name_scope("Hidden_Layers_And_Cell_States_{}".format(n)):
				prev_hidden_layer 			= tf.multiply(L_1_output,tf.tanh(new_cell_state))
				prev_cell_state 			= tf.identity(new_cell_state)

			## Predicted Basket and Actual Basket Differences
		with tf.name_scope("Predictions_{}".format(n)):	
			user_preference 			= tf.sigmoid(tf.matmul(L_1_output,W_2))
			user_new_basket_prediction 	= tf.sigmoid(tf.matmul(user_preference,tf.transpose(product_weights)))
			user_at_time_n1 			= tf.squeeze(tf.slice(input_data,[0,n+1,0,0],[-1,1,-1,1]),[1,3])
			user_next_basket			= tf.round(user_next_basket)
	return user_next_basket

## Run Graph
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,"./model/binary_addition_rnn_lstm_model")

# sess.run(feed_foward_user(),feed_dict = {input_data:})
