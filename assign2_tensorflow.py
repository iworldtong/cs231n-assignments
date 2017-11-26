from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

import config as cfg



def main(data_set="cifar10"):

	if data_set == "cifar10":
		# load cifar10 data set
		train_images, train_labels, val_images, val_labels, test_images, test_labels = cfg.load_cifar10_with_divided()
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

	# clear old variables
	tf.reset_default_graph()

	# setup input
	X = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.int64, [None])
	is_training = tf.placeholder(tf.bool)


	def complex_model(X, y, is_training):
		conv_W1 = tf.get_variable("conv_W1", shape=[7, 7, 3, 32])
		conv_b1 = tf.get_variable("conv_b1", shape=[32])
		gamma = tf.get_variable("gamma", shape=[32], initializer=tf.ones_initializer)
		beta  = tf.get_variable("beta", shape=[32], initializer=tf.zeros_initializer)
		W1 = tf.get_variable("W1", shape=[1152, 1024])
		b1 = tf.get_variable("b1", shape=[1024])
		W2 = tf.get_variable("W2", shape=[1024, 10])
		b2 = tf.get_variable("b2", shape=[10])

		BN_decay = 0.5
		moving_mean = tf.get_variable('moving_mean',  
                                shape=[32],  
                                initializer=tf.zeros_initializer,  
                                trainable=False)  
		moving_var = tf.get_variable('moving_variance',  
                                shape=[32],  
                                initializer=tf.ones_initializer,  
                                trainable=False)

		out = tf.nn.conv2d(X, conv_W1, strides=[1,2,2,1], padding='VALID') + conv_b1
		out = tf.nn.relu(out)

		mean, var = tf.nn.moments(out, [0,1,2])
		train_mean = tf.assign(moving_mean, moving_mean * BN_decay + mean * (1 - BN_decay))
		train_var = tf.assign(moving_var, moving_var * BN_decay + var * (1 - BN_decay))
		with tf.control_dependencies([train_mean, train_var]):
			mean, var = tf.cond(is_training, lambda: (mean, var),  lambda: (moving_mean, moving_var))
			out = tf.nn.batch_normalization(out, mean, var, beta, gamma, 1e-5)

		out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
		out = tf.reshape(out, [-1, 1152])
		out = tf.nn.relu(tf.matmul(out, W1) + b1)
		out = tf.matmul(out, W2) + b2
		return out

	y_out = complex_model(X, y, is_training)
	
	# define loss
	total_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y, 10), logits=y_out)
	mean_loss = tf.reduce_mean(total_loss)

	# define optimizer
	optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(mean_loss)

	def run(session, predict, loss_val, Xd, yd,
			epochs=1, batch_size=64, print_every=100,
			training=None, plot_losses=False):
			# have tensorflow compute accuracy
		correct_prediction = tf.equal(tf.argmax(predict,1), y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# shuffle indicies
		train_indicies = np.arange(Xd.shape[0])
		np.random.shuffle(train_indicies)

		training_now = training is not None
		    
		# setting up variables we want to compute (and optimizing)
		# if we have a training function, add that to things we compute
		variables = [mean_loss,correct_prediction,accuracy]
		if training_now:
			variables[-1] = training
		    
		# counter 
		iter_cnt = 0
		for e in range(epochs):
			# keep track of losses and accuracy
			correct = 0
			losses = []
			# make sure we iterate over the dataset once
			for i in range(int(np.ceil(Xd.shape[0]/batch_size))):
				# generate indicies for the batch
				start_idx = (i*batch_size)%Xd.shape[0]
				idx = train_indicies[start_idx:start_idx+batch_size]
		            
				# create a feed dictionary for this batch
				feed_dict = {X: Xd[idx,:],
							 y: yd[idx],
							 is_training: training_now }
				# get batch size
				actual_batch_size = yd[idx].shape[0]
		            
				# have tensorflow compute loss and correct predictions
				# and (if given) perform a training step
				loss, corr, _ = session.run(variables, feed_dict=feed_dict)
		            
				# aggregate performance stats
				losses.append(loss*actual_batch_size)
				correct += np.sum(corr)

				# print every now and then
				if training_now and (iter_cnt % print_every) == 0:
					print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
								.format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
				iter_cnt += 1
			total_correct = correct/Xd.shape[0]
			total_loss = np.sum(losses)/Xd.shape[0]
			print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
				.format(total_loss,total_correct,e+1))
			if plot_losses:
				plt.plot(losses)
				plt.grid(True)
				plt.title('Epoch {} Loss'.format(e+1))
				plt.xlabel('minibatch number')
				plt.ylabel('minibatch loss')
				plt.show()
		return total_loss,total_correct
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Training')
		run(sess,y_out,mean_loss,train_images,train_labels,1,64,100,optimizer,True)
		print('Validation')
		run(sess,y_out,mean_loss,val_images,val_labels,1,64)

if __name__ == '__main__':
	main()