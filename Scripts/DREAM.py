import tensorflow as tf
import numpy as np

## Load input basket data
in_data = np.load('..\\Processed_Data\\instacart_data_input.npy')
in_dims = np.load('..\\Processed_Data\\input_dims.npy')

## Input dimension sizes
number_of_products 			= in_dims[2]
number_of_product_features 	= in_dims[3]
number_of_time_steps 		= in_dims[1]
number_of_users 			= in_dims[0]

## Ranks for Each Layer
L_1_nodes 		= number_of_product_features
user_features 	= number_of_product_features

## Weight Initialization for LSTM
W_1_i 	= tf.Variable(tf.truncated_normal([number_of_product_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_input")
W_1_c 	= tf.Variable(tf.truncated_normal([number_of_product_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_candidate")
W_1_f 	= tf.Variable(tf.truncated_normal([number_of_product_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_forget")
W_1_o 	= tf.Variable(tf.truncated_normal([number_of_product_features,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_output")
W_h_i 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_input")
W_h_c 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_candidate")
W_h_f 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_forget")
W_h_o 	= tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_output")
# W_cell_state = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_cell_state")
W_2 	= tf.Variable(tf.truncated_normal([L_1_nodes,user_features],stddev = 0.5, mean = 0),name = "W_2_User_Preference")

## Product Feature Initialization
product_weights = tf.Variable(tf.truncated_normal([number_of_products,number_of_product_features],stddev = 0.5,mean = 0),name = "product_weights")

## Placeholders to run graph
input_data			= tf.placeholder(tf.float32,in_dims)
prev_cell_state 	= tf.zeros([number_of_users,L_1_nodes])
prev_hidden_layer 	= tf.zeros([number_of_users,L_1_nodes])
cost 				= tf.reduce_sum(tf.zeros([1]))

for n in range(number_of_time_steps - 1):
	## Prepare input using max of each feature over last n baskets
	user_baskets_0_to_n 		= tf.slice(input_data,[0,0,0,0],[-1,n+1,-1,-1])
	user_baskets_0_to_n 		= tf.multiply(user_baskets_0_to_n,product_weights)
	user_basket_representation 	= tf.reduce_max(tf.reduce_max(user_baskets_0_to_n,2),1)
	
	## Basic LSTM Cell setup
	forget_gate 				= tf.exp(tf.add(tf.matmul(user_basket_representation,W_1_f),tf.matmul(prev_hidden_layer,W_h_f)))
	input_gate 					= tf.exp(tf.add(tf.matmul(user_basket_representation,W_1_i),tf.matmul(prev_hidden_layer,W_1_i)))
	candidate_value 			= tf.tanh(tf.add(tf.matmul(user_basket_representation,W_1_c),tf.matmul(prev_hidden_layer,W_h_c)))
	new_cell_state 				= tf.add(tf.multiply(input_gate,candidate_value),tf.multiply(forget_gate,prev_cell_state))
	
	## Generated User Preference
	L_1_output 					= tf.exp(tf.add(tf.matmul(user_basket_representation,W_1_o),tf.matmul(prev_hidden_layer,W_h_o)))
	user_preference 			= tf.exp(tf.matmul(L_1_output,W_2))
	## Save Cell States and Hidden Layer for next Time Step
	prev_hidden_layer 			= tf.multiply(L_1_output,tf.tanh(new_cell_state))
	prev_cell_state 			= tf.identity(new_cell_state)

	## Predicted Basket and Actual Basket Differences
	user_new_basket_prediction 	= tf.exp(tf.matmul(user_preference,tf.transpose(product_weights)))
	user_at_time_n1 			= tf.squeeze(tf.slice(input_data,[0,n+1,0,0],[-1,1,-1,1]))

	## Cost function sum(Square(Y_u_i_pred - Y_u_i_actual))
	cost 						= tf.add(cost,tf.reduce_sum(tf.square(tf.subtract(user_new_basket_prediction,user_at_time_n1))))
	

## Training Setup
Alpha 			= 0.01
decay_steps 	= 500
decay_rate 		= 0.096
global_step 	= tf.Variable(0,trainable = False,name = "global_step")
learning_rate 	= tf.train.exponential_decay(Alpha,global_step,decay_steps,decay_rate,staircase = True)
train_step 		= tf.train.AdamOptimizer(learning_rate).minimize(cost)


## Run Graph
sess 	= tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

## Iterate training
for n in range(100):
	# if(n % 10):
		# print(sess.run(cost,feed_dict = {input_data:in_data}))
	sess.run(train_step,feed_dict = {input_data:in_data})


writer 	= tf.summary.FileWriter('logs',sess.graph)
saver 	= tf.train.Saver()
saver.save(sess,"./model/DREAM")
print(sess.run(cost,feed_dict = {input_data:in_data}))


