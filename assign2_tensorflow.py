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
	net.reg = 5e-5

	# define optimizer
	optimizer = tf.train.AdamOptimizer(5e-4).minimize(net.loss)

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
    def __init__(self, input_dim=(32,32,3), num_classes=10):
        self.num_classes = num_classes
        self.reg = 5e-4

        self.X = tf.placeholder(tf.float32, [None, *input_dim])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        self.logits = self.build_network(self.X)

        self.softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.y, 10), logits=self.logits)
        self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = self.softmax_loss + self.l2_loss

        self.saver = tf.train.Saver()

    def build_network(self, X):
        out = self.conv_stack(X, filters=128, kernel_size=3)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
        
        out = self.conv_stack(out, filters=128, kernel_size=3)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
        
        out = self.conv_stack(out, filters=256, kernel_size=5)
        out = self.conv_stack(out, filters=256, kernel_size=5)
        out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
        
        out = self.conv_stack(out, filters=128, kernel_size=3)
        out = self.conv_stack(out, filters=128, kernel_size=3)
        
        out = tf.contrib.layers.flatten(out)
        out = tf.layers.dense(out, 1024, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.5, training=self.is_training)
        out = tf.layers.dense(out, 1024, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.5, training=self.is_training)
        out = tf.layers.dense(out, self.num_classes)
        return out
    
    def conv_stack(self, X, filters, kernel_size, activation=tf.nn.relu, padding="SAME"):
        out = tf.layers.conv2d(X, filters=filters, kernel_size=kernel_size, padding=padding, activation=None, \
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg), reuse=False)
        #out = tf.contrib.layers.batch_norm(out,decay=0.9,is_training=self.is_training,zero_debias_moving_mean=True,scale=True,reuse=False)
        out = tf.nn.relu(out)
        return out


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