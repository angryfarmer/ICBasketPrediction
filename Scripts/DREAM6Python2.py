## Need to create way to measure f score
## 	
import time
import tensorflow as tf
import numpy as np
import os
import sys
import imp
data_importer = imp.load_source("data_importer", 'Data_Processing'+os.sep+'data_importer3.py')

orig_out = sys.stdout
outfile = open('out.txt','w')
# sys.stdout = outfile

# print("WTF")

## Load input basket data
data_loader = data_importer.data_importer('..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc',train_batch = 50)

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



user_product_input = tf.placeholder(tf.float32,[None,number_of_time_steps,number_of_products,1])
user_product_DSPO = tf.expand_dims(tf.placeholder(tf.float32,[None,number_of_time_steps]),-1)
user_sequence_length = tf.placeholder(tf.int32,[None])
user_ids = tf.placeholder(tf.int32,[None])



product_weights = tf.Variable(tf.truncated_normal([1,1,number_of_products,number_of_product_features],stddev = 0.5,mean = 0),name = "product_weights")
product_DSPO_weights = tf.Variable(tf.truncated_normal([1,1,number_of_products],stddev = 0.5,mean = 0),name = "product_weights")

product_tiled_inputs = tf.tile(product_weights,(tf.shape(user_product_input)[0],number_of_time_steps,1,1))
product_DSPO_inputs = tf.expand_dims(tf.multiply(user_product_DSPO,product_DSPO_weights),-1)
product_inputs = tf.multiply(user_product_input,tf.concat([product_tiled_inputs,product_DSPO_inputs],3))

feature_dim = tf.reduce_prod(tf.shape(product_inputs)[2:])
product_inputs = tf.reshape(product_inputs,[-1,-1,feature_dim])

user_DSPO_weights = tf.Variable(tf.truncated_normal([number_of_users + 1],stddev = 0.5, mean = 0))

user_subset_weights = tf.expand_dims(tf.expand_dims(tf.nn.embedding_lookup(user_DSPO_weights,user_ids),-1),-1)
user_DSPO_inputs = tf.multiply(user_product_DSPO,user_subset_weights)

inputs = tf.concat([user_DSPO_inputs,product_inputs],2)

keep_prob = 1.0
lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(L_1_nodes),output_keep_prob = keep_prob)
outputs, output_state = tf.nn.dynamic_rnn(inputs = inputs,cell = lstm_cell,sequence_length = number_of_time_steps,dtype = tf.float32)

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
		in_sample = in_sample
		DSPO_sample = DSPO_sample
		print(sess.run(cost,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample}))

def aggregate_error():
	err = 0
	data_loader.reset_to_head()
	while(not data_loader.end_of_file):
		start = time.time()
		in_sample,DSPO_sample,users = data_loader.next_training_sample()
		if(in_sample.size > 0):
			in_sample = in_sample
			DSPO_sample = DSPO_sample
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
				in_sample = in_sample
				DSPO_sample = DSPO_sample
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
	val_data_loader = data_importer.data_importer('..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc',load_batch = 30,include_val = True)
	estimation_points	= 0
	true_positives	= 0
	p_positives = 0

	while(not val_data_loader.end_of_file):
		n += 1
		# print(n)
		# print("Load Index: {},EOF: {}".format(val_data_loader.current_load_index,val_data_loader.end_of_file))
		in_sample,DSPO_sample,users = val_data_loader.next_training_sample()

		if(in_sample.size > 0):
			in_sample = in_sample
			DSPO_sample = DSPO_sample
			correct += sess.run(number_of_correct,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			false_positives += sess.run(number_of_false_positives,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			false_negatives += sess.run(number_of_false_negatives,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			estimation_points 	+= sess.run(tf.reduce_sum(ordered_items),feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			true_positives	+= sess.run(actual_positives,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
			p_positives += sess.run(predicted_positives,feed_dict = {input_data:in_sample,input_DSPO:DSPO_sample})
	print("Correct: {}, False Positives: {}, False Negatives: {}, Data Points: {}, True Positives: {}, Predicted Positives: {}".format(correct,false_positives,false_negatives,estimation_points,true_positives,p_positives))
	# print(n)
	# print(val_data_loader.h5f_train.shape)

print("Done Loading Graph")

start = time.time()
train_graph(500,10)
end = time.time()
print("Time Elapsed for Training: {}".format(end - start))

print(aggregate_error())
# val_error()
# print_error()
