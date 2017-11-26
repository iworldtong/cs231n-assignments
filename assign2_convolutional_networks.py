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

def test(data):
	model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, use_batchnorm=False)

	solver = Solver(model, data,
	                num_epochs=1, batch_size=50,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 1e-3,
	                },
	                verbose=True, print_every=20)
	solver.train()

	grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
	plt.imshow(grid.astype('uint8'))
	plt.axis('off')
	plt.gcf().set_size_inches(5, 5)

	plt.show()

def main(data_set="cifar10"):

	if data_set == "cifar10":
		# load cifar10 data set
		train_images, train_labels, val_images, val_labels, test_images, test_labels = cfg.load_cifar10()		
		classes = cfg.CIFAR10_classes
	elif data_set == "mnist":
		mnist = input_data.read_data_sets(cfg.MNIST_PATH)
		train_images = mnist.train.images
		train_labels = mnist.train.labels
		test_images = mnist.test.images
		test_labels = mnist.test.labels
		classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	train_images = train_images.reshape(-1, 3, 32, 32)
	val_images = val_images.reshape(-1, 3, 32, 32)
	test_images = test_images.reshape(-1, 3, 32, 32)

	data = {
		  'X_train': train_images,
		  'y_train': train_labels,
		  'X_val': val_images,
		  'y_val': val_labels,
		}

	np.random.seed(231)

	test(data)


class ThreeLayerConvNet(object):
	"""
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
	"""
	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
				 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
				 use_batchnorm=False, dtype=np.float32):
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
		self.use_batchnorm = use_batchnorm

		# network. Weights should be initialized from a Gaussian with standard 
		# deviation equal to weight_scale; biases should be initialized to zero.  
		# All weights and biases should be stored in the dictionary self.params.   
		# Store weights and biases for the convolutional layer using the keys 'W1' 
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   
		# of the output affine layer.                                            
		self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
		self.params['b1'] = np.zeros(num_filters)
		self.flatten_dim  = num_filters * np.power(((input_dim[-1] - 1) // 2 + 1), 2)
		self.params['W2'] = weight_scale * np.random.randn(self.flatten_dim, hidden_dim)
		self.params['b2'] = np.zeros(hidden_dim)
		self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
		self.params['b3'] = np.zeros(num_classes)

		self.bn_params = []
		if self.use_batchnorm:
			self.bn_params = [{'mode': 'train'}]
			self.params['gamma1'] = np.ones(num_filters)
			self.params['beta1'] = np.zeros(num_filters)


		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.
		"""
		mode = 'test' if y is None else 'train'
		if self.use_batchnorm:
			gamma1, beta1 = self.params['gamma1'], self.params['beta1']
			for bn_param in self.bn_params:
				bn_param['mode'] = mode
			
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']

		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		
		cache_hist = []
		out_hist = []
		# forward pass 

		# conv - (batchnorm) - relu
		if self.use_batchnorm:
			out, cache = conv_bn_relu_forward(X, W1, b1, gamma1, beta1, conv_param, self.bn_params[0])
		else:
			out, cache = conv_relu_forward(X, W1, b1, conv_param)
		cache_hist.append(cache)
		out_hist.append(out)	

		# 2x2 max pool
		out, cache = max_pool_forward_fast(out, pool_param)
		cache_hist.append(cache)
		out_hist.append(out)
		
		# affine - relu
		#out = out.reshape(X.shape[0], self.flatten_dim)
		out, cache = affine_relu_forward(out, W2, b2)
		cache_hist.append(cache)
		out_hist.append(out)
		
		# affine - softmax
		out, cache = affine_forward(out, W3, b3)
		out_shift = np.exp(out - np.max(out, axis=1, keepdims=True))
		scores = out_shift / np.sum(out_shift, axis=1, keepdims=True)

		if y is None:
			return scores

		# calc loss with L2
		loss, dout = softmax_loss(out, y)
		loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2)+ np.sum(W3**2))
		
		# calc gradiants
		grads = {}
		dout, grads['W3'], grads['b3'] = affine_backward(dout, cache)
		dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache_hist[2])
		dout = max_pool_backward_fast(dout, cache_hist[1])
		if self.use_batchnorm:
			dout, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(dout, cache_hist[0])
		else:
			dout, grads['W1'], grads['b1'] = conv_relu_backward(dout, cache_hist[0])	

		grads['W3'] += self.reg * W3
		grads['W2'] += self.reg * W2
		grads['W1'] += self.reg * W1


		return loss, grads


if __name__ == '__main__':
	main()