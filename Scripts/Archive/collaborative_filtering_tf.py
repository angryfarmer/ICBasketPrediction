import tensorflow as tf
import numpy as np
import pandas as pd 

rank = 6
num_users = 20000
num_products = 20000
num_orders = 200000


# Need to mirror the order numbers
# user_indicies = 
# product_indicies = 

user_weights = tf.Variable(tf.truncated_normal([num_products,rank],stddev = 0.2, mean = 0))
product_weights = tf.Variable(tf.truncated_normal([num_products,rank],stddev = 0.2, mean = 0))

order_user_weights = tf.Gather(user_weights,user_indicies)
order_product_weights = tf.Gather(product_weights,product_indicies)

z = tf.Multiply(order_user_weights,order_product_weights)
g = tf.reciprocal(tf.add(tf.ones(tf.shape(z)),tf.exp(tf.scalar_mul(-1,z))))
h = tf.


sess = tf.InteractiveSession()
print(sess.run(c))

