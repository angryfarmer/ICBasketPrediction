## Need to create way to measure f score
## 	
import time
import tensorflow as tf
import numpy as np
import os
import sys
import imp
data_importer = imp.load_source("data_importer", 'Data_Processing'+os.sep+'data_importer2.py')

orig_out = sys.stdout
outfile = open('out.txt','w')
# sys.stdout = outfile

# print("WTF")

## Load input basket data
data_loader = data_importer.data_importer('..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc',load_batch = 30,train_batch = 20)

## Input dimension sizes
in_dims						= np.load('..'+os.sep+'Processed_Data'+os.sep+'input_dims.npy')
number_of_products 			= in_dims[2]
number_of_data_features 	= in_dims[3]
data_time_steps 			= in_dims[1]
number_of_time_steps 		= data_time_steps
# number_of_time_steps 		= 15
data_time_step_start 		= number_of_time_steps - 10 - 1
number_of_users 			= in_dims[0]

# print(in_dims)
## Ranks for Each Layer
number_of_user_based_features 	= 1
number_of_product_features 		= 5
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
# W_cell_state = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_cell_state")
W_2 	= tf.Variable(tf.truncated_normal([L_1_nodes,number_of_product_features],stddev = 0.5, mean = 0),name = "W_2_User_Preference")

## Product Feature Initialization
product_weights = tf.Variable(tf.truncated_normal([number_of_products,number_of_product_features],stddev = 0.5,mean = 0),name = "product_weights")

## Placeholders to run graph
input_data						= tf.placeholder(tf.float32,[None,number_of_time_steps,number_of_products,1])
input_DSPO						= tf.placeholder(tf.float32,[None,number_of_time_steps])
prev_cell_state 				= tf.zeros([tf.shape(input_data)[0],L_1_nodes])
prev_hidden_layer 				= tf.zeros([tf.shape(input_data)[0],L_1_nodes])
cost 							= tf.reduce_sum(tf.zeros([1]))
prev_user_basket_representation = tf.zeros([tf.shape(input_data)[0],number_of_product_features])
prev_ordered_items 				= tf.zeros([tf.shape(input_data)[0],number_of_products])
for n in range(number_of_time_steps - data_time_step_start - 1):
	with tf.name_scope("Time_Step_{}".format(n)):
		## Prepare input using max of each feature over last n baskets
		with tf.name_scope("Basket_Representation_{}".format(n)):
			## We are only estimating repeat purchases
			if( n == 0):
				ordered_items 					= tf.squeeze(tf.slice(input_data,[0,n,0,0],[-1,data_time_step_start+1,-1,1]),[3])
				ordered_items 					= tf.concat([ordered_items,tf.expand_dims(prev_ordered_items,1)],1)
				ordered_items 					= tf.reduce_max(ordered_items,[1])
				prev_ordered_items 				= tf.identity(ordered_items)

				user_baskets_0_to_n 			= tf.slice(input_data,[0,n,0,0],[-1,data_time_step_start + 1,-1,1])
				user_baskets_0_to_n 			= tf.concat([user_baskets_0_to_n] * number_of_product_features,3)
				user_baskets_0_to_n 			= tf.multiply(user_baskets_0_to_n,product_weights)
				user_baskets_0_to_n 			= tf.reduce_max(user_baskets_0_to_n,2)
				user_baskets_0_to_n				= tf.concat([user_baskets_0_to_n,tf.expand_dims(prev_user_basket_representation,1)],1)
			else:
				ordered_items 					= tf.squeeze(tf.slice(input_data,[0,data_time_step_start+n,0,0],[-1,1,-1,1]),[3])
				ordered_items 					= tf.concat([ordered_items,tf.expand_dims(prev_ordered_items,1)],1)
				ordered_items 					= tf.reduce_max(ordered_items,[1])
				prev_ordered_items 				= tf.identity(ordered_items)

				user_baskets_0_to_n 			= tf.slice(input_data,[0,data_time_step_start+n,0,0],[-1,1,-1,1])
				user_baskets_0_to_n 			= tf.concat([user_baskets_0_to_n] * number_of_product_features,3)
				user_baskets_0_to_n 			= tf.multiply(user_baskets_0_to_n,product_weights)
				user_baskets_0_to_n 			= tf.reduce_max(user_baskets_0_to_n,2)
				user_baskets_0_to_n				= tf.concat([user_baskets_0_to_n,tf.expand_dims(prev_user_basket_representation,1)],1)
			
			user_basket_representation 		= tf.reduce_max(user_baskets_0_to_n,1)
			user_factors 					= tf.slice(input_DSPO,[0,data_time_step_start+n+1],[-1,1])
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
			user_preference 			= tf.sigmoid(tf.matmul(L_1_output,W_2))
		## Save Cell States and Hidden Layer for next Time Step
		with tf.name_scope("Hidden_Layers_And_Cell_States_{}".format(n)):
			prev_hidden_layer 			= tf.multiply(L_1_output,tf.tanh(new_cell_state))
			prev_cell_state 			= tf.identity(new_cell_state)

		## Predicted Basket and Actual Basket Differences. user X number_of_products matrix
		with tf.name_scope("Predictions_{}".format(n)):	
			user_new_basket_prediction 	= tf.sigmoid(tf.matmul(user_preference,tf.transpose(product_weights)))
			user_at_time_n1 			= tf.squeeze(tf.slice(input_data,[0,data_time_step_start+n+1,0,0],[-1,1,-1,1]),[1,3])

		## Actual Basket Output
		with tf.name_scope("Prediction_{}".format(n)):
			user_next_basket			= tf.round(user_new_basket_prediction)
			user_next_basket			= tf.multiply(user_next_basket,ordered_items)

		## Cost function sum(Square(Y_u_i_pred - Y_u_i_actual))
		with tf.name_scope("Cost_{}".format(n)):
			cost 						= tf.add(cost,tf.reduce_sum(tf.abs(tf.multiply(ordered_items,tf.subtract(user_new_basket_prediction,user_at_time_n1)))))

actual_positives = tf.reduce_sum(tf.multiply(user_at_time_n1,ordered_items))
predicted_positives = tf.reduce_sum(user_next_basket)
prediction_errors = tf.subtract(user_next_basket,user_at_time_n1)
pred_shape = tf.shape(prediction_errors)
number_of_correct = tf.reduce_sum(tf.multiply(tf.where(tf.equal(prediction_errors,tf.zeros(pred_shape)),tf.ones(pred_shape),tf.zeros(pred_shape)),ordered_items))
number_of_false_positives = tf.reduce_sum(tf.multiply(tf.where(tf.equal(prediction_errors,tf.ones(pred_shape)),tf.ones(pred_shape),tf.zeros(pred_shape)),ordered_items))
number_of_false_negatives = tf.reduce_sum(tf.multiply(tf.where(tf.equal(prediction_errors,tf.scalar_mul(-1,tf.ones(pred_shape))),tf.ones(pred_shape),tf.zeros(pred_shape)),ordered_items))

precision_numerator = tf.reduce_sum(tf.multiply(actual_positives,predicted_positives))
precision_denominator = tf.reduce_sum(actual_positives)
recall_numerator = tf.reduce_sum(tf.multiply(actual_positives,predicted_positives))
recall_denominator = tf.reduce_sum(actual_positives)
# precision = tf.reduce_sum(tf.multiply(actual_positives,predicted_positives))/tf.reduce_sum(predicted_positives)
# recall = tf.reduce_sum(tf.multiply(actual_positives,predicted_positives))/tf.reduce_sum(actual_positives)
# f1_score = tf.scalar_mul(2,tf.multiply(precision,recall) / tf.add(precision,recall))

## Training Setup
Alpha 			= 0.01
decay_steps 	= 500000
decay_rate 		= 0.096
global_step 	= tf.Variable(0,trainable = False,name = "global_step")
learning_rate 	= tf.train.exponential_decay(Alpha,global_step,decay_steps,decay_rate,staircase = True)
optimizer 		= tf.train.AdamOptimizer(learning_rate)
grads_and_vars 	= optimizer.compute_gradients(cost)
train_step 		= optimizer.apply_gradients(grads_and_vars)


## Run Graph
load_graph = True
if(load_graph):
	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess,'..'+os.sep+'model'+os.sep+'DREAM')
else:
	sess 	= tf.InteractiveSession()
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

def print_error():
	in_sample,DSPO_sample,users = data_loader.next_training_sample()
	if(in_sample.size > 0):
		print(sess.run(cost,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample}))

def aggregate_error():
	err = 0
	data_loader.reset_to_head()
	while(not data_loader.end_of_file):
		start = time.time()
		in_sample,DSPO_sample,users = data_loader.next_training_sample()
		if(in_sample.size > 0):
			err += sess.run(cost,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
		print("Current Indicies: {}, Err: {}, Time Elapsed: {}".format(data_loader.index,err,time.time()-start))
	return err

def train_graph(cycles,print_cycle):		
	## Iterate training
	saver 	= tf.train.Saver()
	for n in range(cycles):
		# print(n)
		if((n % print_cycle) == 0):
		# if(true):
			print("Train Step: {}, Error = {}".format(n,aggregate_error()))
		data_loader.reset_to_head()
		while(not data_loader.end_of_file):
			start = time.time()
			in_sample,DSPO_sample,users = data_loader.next_training_sample()
			end = time.time()
			# print("Cycle: {}, Load Sample Time: {}".format(n,end-start))
			if(in_sample.size > 0):
				start = time.time()
				sess.run(train_step,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
				end = time.time()
				# if(data_loader.index + 1 % 100 == 0):
				print("Train Step Time: {}".format(end-start))
		saver.save(sess,'..'+os.sep+'model'+os.sep+'DREAM')
		sys.stdout = orig_out
		print("Cycle {} Completed".format(n))
		sys.stdout = outfile
	writer 	= tf.summary.FileWriter('logs',sess.graph)
	saver.save(sess,'..'+os.sep+'model'+os.sep+'DREAM')	
	print("Final Error: {}".format(aggregate_error()))


def val_error():
	n = 0
	correct = 0
	false_positives = 0
	false_negatives = 0
	val_data_loader = data_importer.data_importer('..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc',train_batch = 10,include_val = True)
	estimation_points	= 0
	true_positives	= 0
	p_positives = 0
	precision_num = 0
	precision_den = 0
	recall_num = 0
	recall_den = 0

	while(not val_data_loader.end_of_file):
		n += 1
		in_sample,DSPO_sample,users = val_data_loader.next_training_sample()

		if(in_sample.size > 0):
			precision_num += sess.run(precision_numerator,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			precision_den += sess.run(precision_denominator,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			recall_num += sess.run(recall_numerator,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			recall_den += sess.run(recall_denominator,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
		print(n)
		print(users)
	print("Precision: {},Recall: {}".format(precision_num/precision_den,recall_num/recall_den))


print("Done Loading Graph")

start = time.time()
# train_graph(500,10)
end = time.time()
print("Time Elapsed for Training: {}".format(end - start))

# print(aggregate_error())
val_error()
# print_error()
# print(aggregate_error())