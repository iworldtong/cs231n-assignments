from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

import config as cfg
from utils.layers import *
from utils.layer_utils import *
from utils.fast_layers import *
from utils.vis_utils import *
from utils.gradient_check import *
from utils.solver import *


def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def imshow_noax(img, normalize=True):
	""" Tiny helper to show images as uint8 and remove axis labels """
	if normalize:
		img_max, img_min = np.max(img), np.min(img)
		img = 255.0 * (img - img_min) / (img_max - img_min)
	if len(img.shape) == 2: 
		out_img = img[:, :, np.newaxis].repeat([3], axis=2)
	else: out_img = img
	plt.imshow(out_img.astype('uint8'))
	plt.gca().axis('off')



def main(data_set="cifar10"):

	if data_set == "cifar10":
		# load cifar10 data set
		train_images, train_labels, test_images, test_labels = cfg.load_cifar10()
		train_images = train_images.astype(np.float64)
		test_images = test_images.astype(np.float64)
		classes = cfg.CIFAR10_classes
		train_images /= 255.
		test_images /= 255.
	elif data_set == "mnist":
		mnist = input_data.read_data_sets(cfg.MNIST_PATH)
		train_images = mnist.train.images
		train_labels = mnist.train.labels
		test_images = mnist.test.images
		test_labels = mnist.test.labels
		classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



	np.random.seed(231)
	x = np.random.randn(2, 3, 8, 8)
	w = np.random.randn(3, 3, 3, 3)
	b = np.random.randn(3,)
	dout = np.random.randn(2, 3, 8, 8)
	conv_param = {'stride': 1, 'pad': 1}

	out, cache = conv_relu_forward(x, w, b, conv_param)
	dx, dw, db = conv_relu_backward(dout, cache)

	dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
	dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
	db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)

	print('Testing conv_relu:')
	print('dx error: ', rel_error(dx_num, dx))
	print('dw error: ', rel_error(dw_num, dw))
	print('db error: ', rel_error(db_num, db))


class ThreeLayerConvNet(object):
	"""
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
	"""
	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
				 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
				 dtype=np.float32):
		"""
		Initialize a new network.
		
		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Size of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
		  of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		# network. Weights should be initialized from a Gaussian with standard 
		# deviation equal to weight_scale; biases should be initialized to zero.  
		# All weights and biases should be stored in the dictionary self.params.   
		# Store weights and biases for the convolutional layer using the keys 'W1' 
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   
		# of the output affine layer.                                              
		pass

		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.
		"""
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']

		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None
		
		# forward pass 

		pass

		if y is None:
			return scores

		loss, grads = 0, {}
		# The backward pass for the three-layer convolutional net. 
		# Don't forget to add L2 regularization!

		pass

		return loss, grads



if __name__ == '__main__':
	main()