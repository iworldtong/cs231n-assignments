from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time, os, json

import config as cfg
from utils.squeezenet import SqueezeNet
from utils.data_utils import *
from utils.image_utils import *



def get_session():
	"""Create a session that dynamically allocates memory."""
	# See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	return session


def show_some_example():
	plt.figure(figsize=(12, 6))
	for i in range(5):
		plt.subplot(1, 5, i + 1)
		plt.imshow(X_raw[i])
		plt.title(class_names[y[i]])
		plt.axis('off')
	plt.gcf().tight_layout()
	plt.show()


def main():

	tf.reset_default_graph()
	sess = get_session()

	SAVE_PATH = os.path.join(cfg.MODEL_PATH, 'squeezenet.ckpt')
	#if not os.path.exists(SAVE_PATH):
	#	raise ValueError("You need to download SqueezeNet!")
	model = SqueezeNet(save_path=SAVE_PATH, sess=sess)


	X_raw, y, class_names = load_imagenet_val(num=5)
	X = np.array([preprocess_image(img) for img in X_raw])

	#show_some_example()
	
	################################
	#         Saliency Maps        #
	################################
	def compute_saliency_maps(X, y, model):
		"""
	    Compute a class saliency map using the model for images X and labels y.

	    Input:
	    - X: Input images, numpy array of shape (N, H, W, 3)
	    - y: Labels for X, numpy of shape (N,)
	    - model: A SqueezeNet model that will be used to compute the saliency map.

	    Returns:
	    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
	    input images.
	    """
		# Compute the score of the correct class for each example.
		# This gives a Tensor with shape [N], the number of examples.
		#
		# Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
		# for computing vectorized losses.
		correct_scores = tf.gather_nd(model.classifier,
									  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
		###############################################################################
		# TODO: Implement this function. You should use the correct_scores to compute #
		# the loss, and tf.gradients to compute the gradient of the loss with respect #
		# to the input image stored in model.image.                                   #
		# Use the global sess variable to finally run the computation.                #
		# Note: model.image and model.labels are placeholders and must be fed values  #
		# when you call sess.run().                                                   #
		###############################################################################
		grads_op = tf.gradients(ys=correct_scores, xs=[model.image])
		grads = sess.run(grads_op, feed_dict={model.image : X, model.labels : y})[0]
		saliency = np.max(grads, 3)
		return saliency 

	def show_saliency_maps(X, y, mask):
		mask = np.asarray(mask)
		Xm = X[mask]
		ym = y[mask]

		saliency = compute_saliency_maps(Xm, ym, model)

		plt.figure('Saliency Maps')
		for i in range(mask.size):

			plt.subplot(2, mask.size, i + 1)
			plt.title(class_names[ym[i]])
			plt.imshow(deprocess_image(Xm[i]))
			plt.axis('off')			
			plt.subplot(2, mask.size, mask.size + i + 1)
			plt.title(mask[i])
			plt.imshow(saliency[i], cmap=plt.cm.gray)
			plt.axis('off')
			plt.gcf().set_size_inches(10, 4)
		plt.show()

	#show_saliency_maps(X, y, np.arange(X.shape[0]))



	################################
	#        Fooling Images        #
	################################
	def make_fooling_image(X, target_y, model):
		"""
	    Generate a fooling image that is close to X, but that the model classifies
	    as target_y.

	    Inputs:
	    - X: Input image, of shape (1, 224, 224, 3)
	    - target_y: An integer in the range [0, 1000)
	    - model: Pretrained SqueezeNet model

	    Returns:
	    - X_fooling: An image that is close to X, but that is classifed as target_y
	    by the model.
	    """
		X_fooling = X.copy()
		learning_rate = 1
		##############################################################################
		# TODO: Generate a fooling image X_fooling that the model will classify as   #
		# the class target_y. Use gradient ascent on the target class score, using   #
		# the model.classifier Tensor to get the class scores for the model.image.   #
		# When computing an update step, first normalize the gradient:               #
		#   dX = learning_rate * g / ||g||_2                                         #
		#                                                                            #
		# You should write a training loop                                           #
		#                                                                            #  
		# HINT: For most examples, you should be able to generate a fooling image    #
		# in fewer than 100 iterations of gradient ascent.                           #
		# You can print your progress over iterations to check your algorithm.       #
		##############################################################################
		target_scores = tf.gather_nd(model.classifier,
									  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
		grads_op = tf.gradients(ys=target_scores, xs=[model.image])
		for i in range(200):
			grads = sess.run(grads_op, feed_dict={model.image : X_fooling, model.labels : [target_y]})[0]
			X_fooling += learning_rate * grads / np.sum(grads ** 2)
		return X_fooling

	def show_fooling_image(X, idx, target_y):
		Xi = X[idx][None]

		X_fooling = make_fooling_image(Xi, target_y, model)

		# Make sure that X_fooling is classified as y_target
		scores = sess.run(model.classifier, {model.image: X_fooling})
		assert scores[0].argmax() == target_y, 'The network is not fooled!'

		# Show original image, fooling image, and difference
		orig_img = deprocess_image(Xi[0])
		fool_img = deprocess_image(X_fooling[0])
		# Rescale 
		plt.subplot(1, 4, 1)
		plt.imshow(orig_img)
		plt.axis('off')
		plt.title(class_names[y[idx]])
		plt.subplot(1, 4, 2)
		plt.imshow(fool_img)
		plt.title(class_names[target_y])
		plt.axis('off')
		plt.subplot(1, 4, 3)
		plt.title('Difference')
		plt.imshow(deprocess_image((Xi-X_fooling)[0]))
		plt.axis('off')
		plt.subplot(1, 4, 4)
		plt.title('Magnified difference (10x)')
		plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
		plt.axis('off')
		plt.gcf().tight_layout()
		plt.show()

	#show_fooling_image(X=X, idx=3, target_y=6)



	################################
	#      Class Visualization     #
	################################
	from scipy.ndimage.filters import gaussian_filter1d
	def blur_image_by_gaussian(X, sigma=1):
		X = gaussian_filter1d(X, sigma, axis=1)
		X = gaussian_filter1d(X, sigma, axis=2)
		return X

	def create_class_visualization(target_y, model, **kwargs):
		"""
	    Generate an image to maximize the score of target_y under a pretrained model.
	    
	    Inputs:
	    - target_y: Integer in the range [0, 1000) giving the index of the class
	    - model: A pretrained CNN that will be used to generate the image
	    
	    Keyword arguments:
	    - l2_reg: Strength of L2 regularization on the image
	    - learning_rate: How big of a step to take
	    - num_iterations: How many iterations to use
	    - blur_every: How often to blur the image as an implicit regularizer
	    - max_jitter: How much to gjitter the image as an implicit regularizer
	    - show_every: How often to show the intermediate result
	    """
		l2_reg = kwargs.pop('l2_reg', 1e-3)
		learning_rate = kwargs.pop('learning_rate', 25)
		num_iterations = kwargs.pop('num_iterations', 100)
		blur_every = kwargs.pop('blur_every', 10)
		max_jitter = kwargs.pop('max_jitter', 16)
		show_every = kwargs.pop('show_every', 25)

		class_name = class_names[target_y]
		X = 255 * np.random.rand(224, 224, 3)
		X = preprocess_image(X)[None]

		########################################################################
		# TODO: Compute the loss and the gradient of the loss with respect to  #
		# the input image, model.image. We compute these outside the loop so   #
		# that we don't have to recompute the gradient graph at each iteration #
		#                                                                      #
		# Note: loss and grad should be TensorFlow Tensors, not numpy arrays!  #
		#                                                                      #
		# The loss is the score for the target label, target_y. You should     #
		# use model.classifier to get the scores, and tf.gradients to compute  #
		# gradients. Don't forget the (subtracted) L2 regularization term!     #
		########################################################################
		correct_scores = tf.gather_nd(model.classifier,
									  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
		loss =  - correct_scores + l2_reg * tf.reduce_sum(model.image ** 2, axis=[1,2,3])
		# gradient of loss with respect to model.image, same size as model.image
		grads_op = tf.gradients(ys=loss, xs=[model.image]) 


		for t in range(num_iterations):
			# Randomly jitter the image a bit; this gives slightly nicer results
			ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
			Xi = X.copy()
			X = np.roll(np.roll(X, ox, 1), oy, 2)

			########################################################################
			# TODO: Use sess to compute the value of the gradient of the score for #
			# class target_y with respect to the pixels of the image, and make a   #
			# gradient step on the image using the learning rate. You should use   #
			# the grad variable you defined above.                                 #
			#                                                                      #
			# Be very careful about the signs of elements in your code.            #
			########################################################################
			grads = sess.run(grads_op, feed_dict={model.image:X, model.labels:[target_y]})[0]
			X -= learning_rate * grads

			# Undo the jitter
			X = np.roll(np.roll(X, -ox, 1), -oy, 2)

			# As a regularizer, clip and periodically blur
			X = np.clip(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
			if t % blur_every == 0:
				X = blur_image_by_gaussian(X, sigma=0.5)

			# Periodically show the image
			if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
				plt.imshow(deprocess_image(X[0]))				
				plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
				plt.gcf().set_size_inches(4, 4)
				plt.axis('off')
				plt.show()
		return X

	def show_class_visual():
		target_y = np.random.randint(1000)
		print(class_names[target_y])
		X = create_class_visualization(target_y, model)

	show_class_visual()





if __name__ == '__main__':
	main()