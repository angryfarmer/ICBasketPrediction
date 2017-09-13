## Need to create way to measure f score
## 	
import time
import tensorflow as tf
import numpy as np
import os
import sys
import imp
data_importer = imp.load_source("data_importer", 'Data_Processing'+os.sep+'data_importer3.py')

data_loader = data_importer.data_importer('..'+os.sep+'Processed_Data'+os.sep+'user_baskets_loc',train_batch = 2)

in_sample,DSPO_sample,users,seq_len = data_loader.next_training_sample()

## Input dimension sizes
in_dims						= np.load('..'+os.sep+'Processed_Data'+os.sep+'input_dims.npy')
number_of_products 			= in_dims[2]
number_of_data_features 	= in_dims[3]
data_time_steps 			= in_dims[1]
number_of_time_steps 		= data_time_steps
# number_of_time_steps 		= 15
data_time_step_start 		= number_of_time_steps - 10 - 1
number_of_users 			= in_dims[0]

products_input = tf.placeholder(tf.float32,[None,number_of_time_steps,number_of_products,1])
sequence_length= tf.placeholder(tf.int32,[None])

ninth_slice = tf.slice(products_input,[0,9,0,0],[1,1,-1,-1])



gather_sequence = tf.concat([tf.expand_dims(tf.range(tf.shape(sequence_length)[0]),1),tf.expand_dims(sequence_length,1)],1)


final_input = tf.expand_dims(tf.gather_nd(products_input,gather_sequence),1)
equivalence = tf.reduce_sum(tf.cast(tf.equal(ninth_slice,tf.slice(final_input,[0,0,0,0],[1,-1,-1,-1])),tf.int32))
sess = tf.Session()
res = sess.run(tf.shape(final_input),feed_dict = {products_input:in_sample,sequence_length: seq_len})

print(sess.run(gather_sequence,feed_dict = {products_input:in_sample,sequence_length: seq_len}))
print(sess.run(equivalence,feed_dict = {products_input:in_sample,sequence_length: seq_len}))
print(res)