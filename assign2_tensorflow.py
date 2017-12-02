from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

import config as cfg
import os


def main(data_set="cifar10"):

	if data_set == "cifar10":
		# load cifar10 data set
		train_images, train_labels, val_images, val_labels, test_images, test_labels = cfg.load_cifar10()
		train_images = train_images.reshape(-1, 32, 32, 3)
		val_images = val_images.reshape(-1, 32, 32, 3)
		test_images = test_images.reshape(-1, 32, 32, 3)
	elif data_set == "mnist":
		mnist = input_data.read_data_sets(cfg.MNIST_PATH)
		train_images = mnist.train.images
		train_labels = mnist.train.labels
		test_images = mnist.test.images
		test_labels = mnist.test.labels
		classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


	# init network
	net = model(input_dim=(32, 32, 3))

	# define optimizer
	optimizer = tf.train.AdamOptimizer(5e-3).minimize(net.loss)

	# run
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		save_name = 'tf.ckpt'
		restore_name = save_name.split('.')[0] + '_final.ckpt'

		print('Training')
		run(sess, net, train_images, train_labels, batch_size=64, \
			epochs=4, optimizer=optimizer, plot_losses=True, \
			save_model=save_name#, load_model=restore_name,
			)

		print('Validation')
		run(sess, net, val_images, val_labels, load_model=restore_name)



class model(object):
	'''
		architectures:
		
	'''
	def __init__(self, input_dim=(32,32,3), num_classes=10):
		# setup input
		self.weight_scale = 1e-2
		self.reg = 1e-4

		self.X = tf.placeholder(tf.float32, [None, *input_dim])
		self.y = tf.placeholder(tf.int64, [None])
		self.is_training = tf.placeholder(tf.bool)

		self.num_fc_hidden1 = 512
		self.W = {
			'W1' : tf.get_variable("fc_W1", shape=[128, self.num_fc_hidden1]),
			'W2' : tf.get_variable("fc_W2", shape=[self.num_fc_hidden1, num_classes]),
		}
		self.b = {
			'b1' : tf.get_variable("fc_b1", shape=[self.num_fc_hidden1]),
			'b2' : tf.get_variable("fc_b2", shape=[num_classes]),
		}

		self.logits = self.build_network(self.X)

		softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.y, 10), logits=self.logits)
		mean_loss = tf.reduce_mean(softmax_loss)
		tf.add_to_collection("losses", mean_loss)
		self.loss = tf.add_n(tf.get_collection("losses"))

		self.saver = tf.train.Saver()

	def build_network(self, x):
		out = self.bn_relu_conv(x, 7, 16, is_training=self.is_training, scope='conv1')

		out = self.dense_block_6(out, 16, is_training=self.is_training, scope='dense_block_1')
		out = self.bottle_neck(out, 16, is_training=self.is_training, scope='bottleneck_1')

		out = self.dense_block_12(out, 32, is_training=self.is_training, scope='dense_block_2')
		out = self.bottle_neck(out, 32, is_training=self.is_training, scope='bottleneck_2')

		out = self.dense_block_6(out, 16, is_training=self.is_training, scope='dense_block_3')

		out = tf.nn.avg_pool(out, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

		out = tf.reshape(out, [-1, 128])

		out = tf.nn.relu(tf.matmul(out, self.W['W1']) + self.b['b1'])
		out = tf.matmul(out, self.W['W2']) + self.b['b2']
		return out

	def dense_block_6(self, x, k, is_training, scope, input_dense=None):
		with tf.variable_scope(scope):
			if input_dense is None:
				input_dense = x
			else:
				input_dense = tf.concat([input_dense, x], len(x.shape) - 1)
			dense1 = self.dense_ceil_layer(x, k, is_training=is_training, scope='dense_ceil_1')
			input_dense = tf.concat([x, dense1], len(x.shape) - 1)
			dense2 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_2')
			input_dense = tf.concat([input_dense, dense2], len(x.shape) - 1)
			dense3 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_3')
			input_dense = tf.concat([input_dense, dense3], len(x.shape) - 1)
			dense4 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_4')
			input_dense = tf.concat([input_dense, dense4], len(x.shape) - 1)
			dense5 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_5')
			input_dense = tf.concat([input_dense, dense5], len(x.shape) - 1)
			dense6 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_6')
			input_dense = tf.concat([input_dense, dense6], len(x.shape) - 1)
			out = input_dense
		return out

	def dense_block_12(self, x, k, is_training, scope, input_dense=None):
		with tf.variable_scope(scope):
			if input_dense is None:
				input_dense = x
			else:
				input_dense = tf.concat([input_dense, x], len(x.shape) - 1)
			dense1 = self.dense_ceil_layer(x, k, is_training=is_training, scope='dense_ceil_1')
			input_dense = tf.concat([x, dense1], len(x.shape) - 1)
			dense2 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_2')
			input_dense = tf.concat([input_dense, dense2], len(x.shape) - 1)
			dense3 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_3')
			input_dense = tf.concat([input_dense, dense3], len(x.shape) - 1)
			dense4 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_4')
			input_dense = tf.concat([input_dense, dense4], len(x.shape) - 1)
			dense5 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_5')
			input_dense = tf.concat([input_dense, dense5], len(x.shape) - 1)
			dense6 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_6')
			input_dense = tf.concat([input_dense, dense6], len(x.shape) - 1)
			dense7 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_7')
			input_dense = tf.concat([input_dense, dense7], len(x.shape) - 1)
			dense8 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_8')
			input_dense = tf.concat([input_dense, dense8], len(x.shape) - 1)
			dense9 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_9')
			input_dense = tf.concat([input_dense, dense9], len(x.shape) - 1)
			dense10 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_10')
			input_dense = tf.concat([input_dense, dense10], len(x.shape) - 1)
			dense11 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_11')
			input_dense = tf.concat([input_dense, dense11], len(x.shape) - 1)
			dense12 = self.dense_ceil_layer(input_dense, k, is_training=is_training, scope='dense_ceil_12')
			input_dense = tf.concat([input_dense, dense12], len(x.shape) - 1)
			out = input_dense
		return out

	def dense_ceil_layer(self, x, num_filter, is_training, scope):
		with tf.variable_scope(scope):
			out = self.bn_relu_conv(x, ksize=1, num_filter=num_filter, is_training=is_training, scope='conv1')
			out = self.bn_relu_conv(out, ksize=3, num_filter=num_filter, is_training=is_training, scope='conv2')
		return out

	def bottle_neck(self, x, num_filter, is_training, scope):
		with tf.variable_scope(scope):
			W = tf.Variable(self.weight_scale*tf.truncated_normal(shape=[3, 3, tf.cast(x.get_shape()[-1], tf.int64), num_filter], dtype=tf.float32), name='conv_W', trainable=True)
			out = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
			out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(self.reg)(W))
		return out

	def bn_relu_conv(self, x, ksize, num_filter, is_training, scope):
		with tf.variable_scope(scope):
			W = tf.Variable(self.weight_scale*tf.truncated_normal(shape=[ksize, ksize, tf.cast(x.get_shape()[-1], tf.int64), num_filter], dtype=tf.float32), name='conv_W', trainable=True)
			out = self.batch_norm_layer(x, is_training, 'bn')
			out = tf.nn.relu(out)
			out = tf.nn.conv2d(out, W, strides=[1,1,1,1], padding='SAME')
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(self.reg)(W))
		return out

	def batch_norm_layer(self, x, is_training, scope):
		with tf.variable_scope(scope):
			beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
			gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
			axises = list(range(len(x.shape) - 1))
			batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=0.5)
			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)
			mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
			normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
		return normed


def run(session, net, X, y,
		epochs=1, batch_size=64, load_model=None, save_model='tf.ckpt',
		print_every=100, optimizer=None, plot_losses=False):
	# compute accuracy by tensorflow
	correct_prediction = tf.equal(tf.argmax(net.logits, 1), net.y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# shuffle indicies
	train_indicies = np.arange(X.shape[0])
	np.random.shuffle(train_indicies)
		    
	is_training = optimizer is not None

	# setting up variables we want to compute (and optimizing)
	# if we have a training function, add that to things we compute
	# or load trained model
	variables = [net.loss, correct_prediction, accuracy]
	if is_training:
		variables[-1] = optimizer
	
	if load_model is not None:		
		net.saver.restore(session, os.path.join(cfg.MODEL_PATH, load_model))

	# counter 
	iter_cnt = 0
	for e in range(epochs):
		# keep track of losses and accuracy
		correct = 0
		losses = []
		# make sure we iterate over the dataset once
		for i in range(int(np.ceil(X.shape[0] / batch_size))):
			# generate indicies for the batch
			start_idx = (i * batch_size) % X.shape[0]
			idx = train_indicies[start_idx : start_idx + batch_size]

			# get batch size
			actual_batch_size = y[idx].shape[0]
	            
			# have tensorflow compute loss and correct predictions
			# and (if given) perform a training step
			loss, corr, _ = session.run(variables, feed_dict={net.X: X[idx,:],
															  net.y: y[idx],
															  net.is_training: is_training,
															}
										)
	            
			# aggregate performance stats
			losses.append(loss * actual_batch_size)
			correct += np.sum(corr)

			# print every now and then
			if is_training and (iter_cnt % print_every) == 0:
				print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
							.format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
			iter_cnt += 1

		# average
		total_correct = correct / X.shape[0]
		total_loss = np.sum(losses) / X.shape[0]

		print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
			.format(total_loss, total_correct, (e + 1)))

	# save model
	if is_training:
		name = save_model.split('.')[0] + '_final.' + save_model.split('.')[1]
		net.saver.save(session, os.path.join(cfg.MODEL_PATH, name))


	if plot_losses:
		plt.plot(losses)
		plt.grid(True)
		plt.title('Training Loss')
		plt.xlabel('minibatch number')
		plt.ylabel('minibatch loss')
		plt.show()

	return total_loss,total_correct



if __name__ == '__main__':
	main()