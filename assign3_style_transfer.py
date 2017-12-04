from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

import config as cfg
from utils.image_utils import *
from utils.squeezenet import SqueezeNet


def get_session():
	"""Create a session that dynamically allocates memory."""
	# See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	return session

def rel_error(x,y):
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
	import scipy
	vnum = int(scipy.__version__.split('.')[1])
	assert vnum >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."





def main():

	check_scipy()

	tf.reset_default_graph() # remove all existing variables in the graph 
	sess = get_session() # start a new Session

	# Load pretrained SqueezeNet model
	SAVE_PATH = os.path.join(cfg.MODEL_PATH, 'squeezenet.ckpt')
	#if not os.path.exists(SAVE_PATH):
	#	raise ValueError("You need to download SqueezeNet!")
	model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

	# Load data for testing
	content_img_test = preprocess_image(load_image(os.path.join(cfg.SRC_PATH, 'tubingen.jpg'), size=192))[None]
	style_img_test = preprocess_image(load_image(os.path.join(cfg.SRC_PATH, 'starry_night.jpg'), size=192))[None]
	answers = np.load('./utils/style-transfer-checks-tf.npz')


	def content_loss(content_weight, content_current, content_original):
		"""
	    Compute the content loss for style transfer.
	    
	    Inputs:
	    - content_weight: scalar constant we multiply the content_loss by.
	    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
	    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
	    
	    Returns:
	    - scalar content loss
		"""
		content_loss = content_weight * tf.reduce_sum(tf.squared_difference(content_current, content_original))
		return content_loss

	def content_loss_test(correct):
		content_layer = 3
		content_weight = 6e-2
		c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img_test})
		bad_img = tf.zeros(content_img_test.shape)
		feats = model.extract_features(bad_img)[content_layer]
		student_output = sess.run(content_loss(content_weight, c_feats, feats))
		error = rel_error(correct, student_output)
		print('Maximum content loss error is {:.3f}'.format(error))

	#content_loss_test(answers['cl_out'])


	def gram_matrix(features, normalize=True):
		"""
	    Compute the Gram matrix from features.
	    
	    Inputs:
	    - features: Tensor of shape (1, H, W, C) giving features for
	      a single image.
	    - normalize: optional, whether to normalize the Gram matrix
	        If True, divide the Gram matrix by the number of neurons (H * W * C)
	    
	    Returns:
	    - gram: Tensor of shape (C, C) giving the (optionally normalized)
	      Gram matrices for the input image.
		"""
		_, H, W, C = features.get_shape()
		features_shift = tf.reshape(tf.transpose(features, [3,0,1,2]), [tf.cast(C, tf.int32), -1])
		gram = tf.matmul(features_shift, tf.transpose(features_shift))
		if normalize == True:
			gram = tf.divide(gram, tf.reduce_sum(tf.ones_like(features)))
		return gram

	def gram_matrix_test(correct):
		gram = gram_matrix(model.extract_features()[5])
		student_output = sess.run(gram, {model.image: style_img_test})
		error = rel_error(correct, student_output)
		print('Maximum gram matrix error is {:.3f}'.format(error))

	#gram_matrix_test(answers['gm_out'])


	def style_loss(feats, style_layers, style_targets, style_weights):
		"""
	    Computes the style loss at a set of layers.
	    
	    Inputs:
	    - feats: list of the features at every layer of the current image, as produced by
	      the extract_features function.
	    - style_layers: List of layer indices into feats giving the layers to include in the
	      style loss.
	    - style_targets: List of the same length as style_layers, where style_targets[i] is
	      a Tensor giving the Gram matrix the source style image computed at
	      layer style_layers[i].
	    - style_weights: List of the same length as style_layers, where style_weights[i]
	      is a scalar giving the weight for the style loss at layer style_layers[i].
	      
	    Returns:
	    - style_loss: A Tensor contataining the scalar style loss.
		"""
		# Hint: you can do this with one for loop over the style layers, and should
		# not be very much code (~5 lines). You will need to use your gram_matrix function.
		style_loss = 0.0
		for idx, layer in enumerate(style_layers):
			targets_style = style_targets[idx]
			input_style = gram_matrix(feats[layer])
			style_loss += style_weights[idx] * tf.reduce_sum(tf.squared_difference(targets_style, input_style))
		return style_loss

	def style_loss_test(correct):
		style_layers = [1, 4, 6, 7]
		style_weights = [300000, 1000, 15, 3]

		feats = model.extract_features()
		style_target_vars = []
		for idx in style_layers:
			style_target_vars.append(gram_matrix(feats[idx]))
		style_targets = sess.run(style_target_vars,
								{model.image: style_img_test})                    
		s_loss = style_loss(feats, style_layers, style_targets, style_weights)
		student_output = sess.run(s_loss, {model.image: content_img_test})
		error = rel_error(correct, student_output)
		print('Style loss error is {:.3f}'.format(error))

	#style_loss_test(answers['sl_out'])


	def tv_loss(img, tv_weight):
		"""
	    Compute total variation loss.
	    
	    Inputs:
	    - img: Tensor of shape (1, H, W, 3) holding an input image.
	    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
	    
	    Returns:
	    - loss: Tensor holding a scalar giving the total variation loss
	      for img weighted by tv_weight.
		"""
		# Your implementation should be vectorized and not require any loops!
		W_tv = tf.reduce_sum(tf.squared_difference(img[0,0:-1,1:  ,:], img[0,0:-1,0:-1,:]))
		H_tv = tf.reduce_sum(tf.squared_difference(img[0,1:  ,0:-1,:], img[0,0:-1,0:-1,:]))
		tv_loss = tv_weight * (W_tv + H_tv)
		return tv_loss

	def tv_loss_test(correct):
		tv_weight = 2e-2
		t_loss = tv_loss(model.image, tv_weight)
		student_output = sess.run(t_loss, {model.image: content_img_test})
		error = rel_error(correct, student_output)
		print('Total-variation regularization error is {:.3f}'.format(error))

	#tv_loss_test(answers['tv_out'])



	def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
					   style_layers, style_weights, tv_weight, init_random = False):
		"""Run style transfer!
	    
	    Inputs:
	    - content_image: filename of content image
	    - style_image: filename of style image
	    - image_size: size of smallest image dimension (used for content loss and generated image)
	    - style_size: size of smallest style image dimension
	    - content_layer: layer to use for content loss
	    - content_weight: weighting on content loss
	    - style_layers: list of layers to use for style loss
	    - style_weights: list of weights to use for each layer in style_layers
	    - tv_weight: weight of total variation regularization term
	    - init_random: initialize the starting image to uniform random noise
		"""
		# Extract features from the content image
		content_img = preprocess_image(load_image(content_image, size=image_size))
		feats = model.extract_features(model.image)
		content_target = sess.run(feats[content_layer],
								  {model.image: content_img[None]})

		# Extract features from the style image
		style_img = preprocess_image(load_image(style_image, size=style_size))
		style_feat_vars = [feats[idx] for idx in style_layers]
		style_target_vars = []
		# Compute list of TensorFlow Gram matrices
		for style_feat_var in style_feat_vars:
			style_target_vars.append(gram_matrix(style_feat_var))
		# Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
		style_targets = sess.run(style_target_vars, {model.image: style_img[None]})

		# Initialize generated image to content image

		if init_random:
			img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
		else:
			img_var = tf.Variable(content_img[None], name="image")

		# Extract features on generated image
		feats = model.extract_features(img_var)
		# Compute loss
		c_loss = content_loss(content_weight, feats[content_layer], content_target)
		s_loss = style_loss(feats, style_layers, style_targets, style_weights)
		t_loss = tv_loss(img_var, tv_weight)
		loss = c_loss + s_loss + t_loss

		# Set up optimization hyperparameters
		initial_lr = 3.0
		decayed_lr = 0.1
		decay_lr_at = 180
		max_iter = 300

		# Create and initialize the Adam optimizer
		lr_var = tf.Variable(initial_lr, name="lr")
		# Create train_op that updates the generated image when run
		with tf.variable_scope("optimizer") as opt_scope:
			train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
		# Initialize the generated image and optimization variables		
		opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
		sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
		# Create an op that will clamp the image values when run
		clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

		ck_iter = []
		for i in range(1, max_iter+1):
			if i == 1 or i % 100 == 0 or i == max_iter:
				ck_iter.append(i)

		f, axarr = plt.subplots(1 + int(np.ceil(len(ck_iter)/2)), 2)
		for i in range(axarr.shape[0]):
			for j in range(axarr.shape[1]):
				axarr[i,j].axis('off')
		axarr[0,0].set_title('Content Source Image')
		axarr[0,1].set_title('Style Source Image')
		axarr[0,0].imshow(deprocess_image(content_img))
		axarr[0,1].imshow(deprocess_image(style_img))

		# Hardcoded handcrafted 
		plot_idx = 2
		for t in range(1, max_iter+1):
			# Take an optimization step to update img_var
			sess.run(train_op)
			if t < decay_lr_at:
				sess.run(clamp_image_op)
			if t == decay_lr_at:
				sess.run(tf.assign(lr_var, decayed_lr))
			if t in ck_iter:
				#print('Iteration {}'.format(t))
				img = sess.run(img_var)
				#plt.imshow(deprocess_image(img[0], rescale=True))
				axarr[plot_idx//2, plot_idx%2].imshow(deprocess_image(img[0], rescale=True))
				axarr[plot_idx//2, plot_idx%2].set_title('Iteration {}'.format(t))
				plot_idx += 1
		plt.show()


	params1 = {
		'content_image' : os.path.join(cfg.SRC_PATH, 'tubingen.jpg'),
		'style_image' : os.path.join(cfg.SRC_PATH, 'starry_night.jpg'),
		'image_size' : 192,
		'style_size' : 192,
		'content_layer' : 3,
		'content_weight' : 6e-2,
		'style_layers' : [1, 4, 6, 7],
		'style_weights' : [300000, 1000, 15, 3],
		'tv_weight' : 2e-2,
	}

	params_inv = {
		'content_image' : os.path.join(cfg.SRC_PATH, 'tubingen.jpg'),
		'style_image' :  os.path.join(cfg.SRC_PATH, 'starry_night.jpg'),
		'image_size' : 192,
		'style_size' : 192,
		'content_layer' : 3,
		'content_weight' : 6e-2,
		'style_layers' : [1, 4, 6, 7],
		'style_weights' : [0, 0, 0, 0], # we discard any contributions from style to the loss
		'tv_weight' : 2e-2,
		'init_random': True # we want to initialize our image to be random
	}

	style_transfer(**params1)



if __name__ == '__main__':
	main()