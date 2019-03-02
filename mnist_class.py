import numpy as np
import tensorflow as tf
from chp1 import gen_moon
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from decision import decision_boundary_flcn
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

#Definition of the constants
n_feature = 784
early_stop_step=6
n_output = 10
hd_nuer = [300,100,n_output]
lr=0.01
n_epoch=100
batch_size=8000
mnist = input_data.read_data_sets("/tmp/data/")

#Tensorflow graph construction
tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32,shape=(None,n_feature),name="X")
Y = tf.placeholder(dtype=tf.int32,shape=(None),name="Y")
# hid_init = tf.initializers.random_normal(stddev=0.5)
is_training=tf.placeholder(tf.bool,shape=(),name="is_training")
bn_params={'is_training':is_training,'decay':0.99,'updates_collections:None'}
hidden = fully_connected(X,hd_nuer[0],normalizer_fn=batch_norm,normalizer_params=bn_params)
for i in range(1,len(hd_nuer)-1):
	hidden = fully_connected(hidden,hd_nuer[i],normalizer_fn=batch_norm,normalizer_params=bn_params)
logits = fully_connected(hidden,hd_nuer[len(hd_nuer)-1],activation_fn=None,normalizer_fn=batch_norm,normalizer_params=bn_params)
xentry=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
loss = tf.reduce_sum(xentry,axis=0)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
training_op = optimizer.minimize(loss)
correct=tf.nn.in_top_k(logits,Y,1)
al_in = tf.reduce_mean(tf.cast(correct,tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Tensorflow graph execution
with tf.Session() as sess:
	sess.run(init)
	# # logits_val = X.eval(feed_dict={X:x[1:3,:],Y:y[1:3]})
	# X_batch, y_batch = mnist.train.next_batch(batch_size)
	# print(X_batch)
	best_val=0
	step=0
	ep = 0
	for ep in range(n_epoch):
		for iteration in range(50):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict={is_training:True,X: X_batch, Y: y_batch})
			acc_train = al_in.eval(feed_dict={is_training:False,X: X_batch,Y: y_batch})
			acc_test = al_in.eval(feed_dict={is_training:False,X: mnist.test.images,Y: mnist.test.labels})
			print(step,"\t",ep, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
			if step>early_stop_step:break
			if (acc_test >= best_val):
				best_val=acc_test
				step=0
			else:
				step+=1
		if step>early_stop_step:break
		ep+=1
	save_path=saver.save(sess,"./mnist_class.ckpt")
	sess.close()
