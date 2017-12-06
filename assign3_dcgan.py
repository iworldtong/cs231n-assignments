import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

import config as cfg


# A bunch of utility functions
def show_images(images):
	images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
	sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
	sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

	fig = plt.figure(figsize=(sqrtn, sqrtn))
	gs = gridspec.GridSpec(sqrtn, sqrtn)
	gs.update(wspace=0.05, hspace=0.05)

	for i, img in enumerate(images):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(img.reshape([sqrtimg,sqrtimg]))
	return

def preprocess_img(x):
	return 2 * x - 1.0

def deprocess_img(x):
	return (x + 1.0) / 2.0

def rel_error(x,y):
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
	"""Count the number of parameters in the current TensorFlow graph """
	param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
	return param_count


def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	return session


def sample_noise(batch_size, dim):
	"""Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
	# TODO: sample and return noise
	return tf.random_uniform(shape=[batch_size,dim], minval=-1, maxval=1)

def leaky_relu(x, alpha=0.01):
	"""Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
	# TODO: implement leaky ReLU
	return tf.maximum(alpha*x, x)

def discriminator(x):
	"""Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
	"""
	with tf.variable_scope("discriminator"):
		input_data = tf.reshape(x, (-1, 28, 28, 1))
		out = tf.layers.conv2d(input_data, filters=32, kernel_size=5, activation=leaky_relu, padding="VALID")
		out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
		out = tf.layers.conv2d(out, filters=64, kernel_size=5, activation=leaky_relu, padding="VALID")
		out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
		out = tf.contrib.layers.flatten(out)
		out = tf.layers.dense(out, 1024, activation=leaky_relu, use_bias=True)
		out = tf.layers.dense(out, 1  , use_bias=True)
		logits = out
		return logits

def generator(z):
	"""Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
	"""
	with tf.variable_scope("generator"):
		out = tf.layers.dense(z  , 1024, activation=tf.nn.relu, use_bias=True)
		out = tf.layers.batch_normalization(out, training=True)
		out = tf.layers.dense(out, 6272, activation=tf.nn.relu, use_bias=True)
		out = tf.layers.batch_normalization(out, training=True)
		out = tf.reshape(out, (-1, 7, 7, 128))
		out = tf.layers.conv2d_transpose(out, 64, kernel_size=2, strides=2, activation=tf.nn.relu)
		out = tf.layers.batch_normalization(out, training=True)
		out = tf.layers.conv2d_transpose(out, 1 , kernel_size=2, strides=2, activation=tf.nn.tanh)
		img = out
		return img

def gan_loss(logits_real, logits_fake):
	"""Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
	"""
	# TODO: compute D_loss and G_loss
	a_logits_real = tf.nn.sigmoid(logits_real)
	a_logits_fake = tf.nn.sigmoid(logits_fake)
	D_loss = - tf.reduce_mean(tf.log(a_logits_real)) - tf.reduce_mean(tf.log(1 - a_logits_fake))
	G_loss = - tf.reduce_mean(tf.log(a_logits_fake))
	return D_loss, G_loss


def lsgan_loss(score_real, score_fake):
	"""Compute the Least Squares GAN loss.
    
    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    
          
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
	"""
	# TODO: compute D_loss and G_loss
	D_loss = 0.5 * (tf.reduce_mean(tf.square(score_real - 1)) + tf.reduce_mean(tf.square(score_fake)))
	G_loss = 0.5 * tf.reduce_mean(tf.square(score_fake - 1))
	return D_loss, G_loss


# TODO: create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
	"""Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
	"""
	D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
	G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
	return D_solver, G_solver




def main():

	answers = np.load('./utils/gan-checks-tf.npz')

	mnist = input_data.read_data_sets(cfg.MNIST_PATH, one_hot=False)	

	tf.reset_default_graph()

	# number of images for each batch
	batch_size = 128
	# our noise dimension
	noise_dim = 96

	# placeholder for images from the training dataset
	x = tf.placeholder(tf.float32, [None, 784])
	# random noise fed into our generator
	z = sample_noise(batch_size, noise_dim)
	# generated images
	G_sample = generator(z)

	with tf.variable_scope("") as scope:
		#scale images to be -1 to 1
		logits_real = discriminator(preprocess_img(x))
		# Re-use discriminator weights on new inputs
		scope.reuse_variables()
		logits_fake = discriminator(G_sample)

	# Get the list of variables for the discriminator and generator
	D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
	G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

	# get our solver
	D_solver, G_solver = get_solvers()
	
	# get our loss
	#D_loss, G_loss = gan_loss(logits_real, logits_fake)
	D_loss, G_loss = lsgan_loss(logits_real, logits_fake)

	# setup training steps
	D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
	G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
	D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
	G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

	# a giant helper function
	def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
	              show_every=250, print_every=50, batch_size=128, num_epoch=5):
	    """Train a GAN for a certain number of epochs.
	    
	    Inputs:
	    - sess: A tf.Session that we want to use to run our data
	    - G_train_step: A training step for the Generator
	    - G_loss: Generator loss
	    - D_train_step: A training step for the Generator
	    - D_loss: Discriminator loss
	    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
	    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
	    Returns:
	        Nothing
	    """
	    # compute the number of iterations we need
	    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
	    for it in range(max_iter):
	        # every show often, show a sample result
	        if it % show_every == 0:
	            samples = sess.run(G_sample)
	            fig = show_images(samples[:16])
	            plt.show()
	            print()
	        # run a batch of data through the network
	        minibatch, minbatch_y = mnist.train.next_batch(batch_size)
	        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
	        _, G_loss_curr = sess.run([G_train_step, G_loss])

	        # print loss every so often.
	        # We want to make sure D_loss doesn't go to 0
	        if it % print_every == 0:
	            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
	    
	    print('Final images')
	    samples = sess.run(G_sample)
	    fig = show_images(samples[:16])
	    plt.show()

	with get_session() as sess:
		sess.run(tf.global_variables_initializer())
		run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=1000)
	



if __name__ == '__main__':
	main()