import numpy as np
import tensorflow as tf
from chp1 import gen_moon
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from decision import decision_boundary_flcn
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

#Definition of the constants
n_feature = 28*28
n_output = 10
hd_nuer = [300,100,n_output]
# layer = np.r_[n_feature,hd_nuer]
lr=0.1
n_epoch=20
batch_size=80
mnist = input_data.read_data_sets("/tmp/data/")

#Tensorflow graph construction
tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32,shape=(None,n_feature),name="X")
Y = tf.placeholder(dtype=tf.int32,shape=(None),name="Y")
# hid_init = tf.initializers.random_normal(stddev=0.5)
hidden = fully_connected(X,hd_nuer[0])
for i in range(1,len(hd_nuer)-1):
	hidden = fully_connected(hidden,hd_nuer[i])
logits = fully_connected(hidden,hd_nuer[len(hd_nuer)-1],activation_fn=None)
xentry=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
loss = tf.reduce_sum(xentry,axis=0)
# mse = tf.sqrt(tf.reduce_sum(tf.square(d-hidden),axis=0))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
training_op = optimizer.minimize(loss)
correct=tf.nn.in_top_k(logits,Y,1)
al_in = tf.reduce_mean(tf.cast(correct,tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Tensorflow graph execution
with tf.Session() as sess:
	sess.run(init)
	# logits_val = xentry.eval(feed_dict={X:x[1:3,:],Y:y[1:3]})
	# print(logits_val.shape)
	for ep in range(n_epoch):
		rand_inx = np.random.permutation(len(x))
		x=x[rand_inx]
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
			acc_test = accuracy.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})
		print(ep, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
	save_path=saver.save(sess,"./mnist_class.ckpt")
	sess.close()
