from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from utils.layers import *
from utils.layer_utils import *
from utils.vis_utils import *
from utils.gradient_check import *
from utils.solver import *


def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


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

	np.random.seed(231)
	# Try training a very deep net with batchnorm
	hidden_dims = [100, 100, 100, 100, 100]

	num_train = 1000
	small_data = {
	  'X_train': train_images[:num_train],
	  'y_train': train_labels[:num_train],
	  'X_val': val_images,
	  'y_val': val_labels,
	}

	bn_solvers = {}
	solvers = {}
	weight_scales = np.logspace(-4, 0, num=20)
	for i, weight_scale in enumerate(weight_scales):
	  print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
	  bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
	  model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

	  bn_solver = Solver(bn_model, small_data,
	                  num_epochs=10, batch_size=50,
	                  update_rule='adam',
	                  optim_config={
	                    'learning_rate': 1e-3,
	                  },
	                  verbose=False, print_every=200)
	  bn_solver.train()
	  bn_solvers[weight_scale] = bn_solver

	  solver = Solver(model, small_data,
	                  num_epochs=10, batch_size=50,
	                  update_rule='adam',
	                  optim_config={
	                    'learning_rate': 1e-3,
	                  },
	                  verbose=False, print_every=200)
	  solver.train()
	  solvers[weight_scale] = solver

	  # Plot results of weight scale experiment
	best_train_accs, bn_best_train_accs = [], []
	best_val_accs, bn_best_val_accs = [], []
	final_train_loss, bn_final_train_loss = [], []

	for ws in weight_scales:
	  best_train_accs.append(max(solvers[ws].train_acc_history))
	  bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))
	  
	  best_val_accs.append(max(solvers[ws].val_acc_history))
	  bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))
	  
	  final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
	  bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))
	  
	plt.subplot(3, 1, 1)
	plt.title('Best val accuracy vs weight initialization scale')
	plt.xlabel('Weight initialization scale')
	plt.ylabel('Best val accuracy')
	plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
	plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
	plt.legend(ncol=2, loc='lower right')

	plt.subplot(3, 1, 2)
	plt.title('Best train accuracy vs weight initialization scale')
	plt.xlabel('Weight initialization scale')
	plt.ylabel('Best training accuracy')
	plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
	plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
	plt.legend()

	plt.subplot(3, 1, 3)
	plt.title('Final training loss vs weight initialization scale')
	plt.xlabel('Weight initialization scale')
	plt.ylabel('Final training loss')
	plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
	plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
	plt.legend()
	plt.gca().set_ylim(1.0, 3.5)

	plt.gcf().set_size_inches(10, 15)
	plt.show()


class FullyConnectedNet(object):
	"""
	architecture (L layers) :
		{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
	"""
	def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,\
				 dropout=0, use_batchnorm=False, reg=0.0,\
				 weight_scale=1.0, dtype=np.float32, seed=None):

		self.use_batchnorm = use_batchnorm
		self.use_dropout = dropout > 0
		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}

		self.dim_list = [input_dim, num_classes]
		self.dim_list[1:1] = hidden_dims


		# Initialize the parameters of the network, storing all values in the self.params dictionary. 
		# When using batch normalization, store scale and shift parameters 
		# For the first layer in gamma1 and beta1; for the second layer use gamma2 and beta2, etc. 
		# Scale parameters should be initialized to one and shift parameters should be initialized to zero. 
		for l in range(self.num_layers):
			U = np.sqrt(6.0 / (self.dim_list[l] + self.dim_list[l+1]))		
			#self.params['W' + str(l+1)] = weight_scale * np.random.random((self.dim_list[l], self.dim_list[l+1]))
			self.params['W' + str(l+1)] = 2*weight_scale*U*(np.random.rand(self.dim_list[l], self.dim_list[l+1])-0.5) 
			self.params['b' + str(l+1)] = np.zeros(self.dim_list[l+1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {'mode': 'train', 'p': dropout}
			if seed is not None:
				self.dropout_param['seed'] = seed

		# With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
		self.bn_params = []
		if self.use_batchnorm:
			self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
			for l in range(self.num_layers-1):
				self.params['gamma' + str(l+1)] = np.random.rand(self.dim_list[l+1])
				self.params['beta'  + str(l+1)] = np.random.rand(self.dim_list[l+1])

		# Cast all parameters to the correct datatype
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		if self.use_dropout:
			self.dropout_param['mode'] = mode
		if self.use_batchnorm:
			for bn_param in self.bn_params:
				bn_param['mode'] = mode

		out_hist = []
		cache_hist = []
		# forward
		out = X.copy()
		for l in range(self.num_layers):
			if l < (self.num_layers - 1):	
				bn_cache, drop_cache = None, None
				# dropout
				if self.use_dropout:
					temp_W, drop_cache = dropout_forward(self.params['W'+str(l+1)], self.dropout_param)
				else:
					temp_W = self.params['W'+str(l+1)]
				
				# affine
				out, fc_cache = affine_forward(out, temp_W, self.params['b'+str(l+1)])
				
				# batch normalize
				if self.use_batchnorm:
					out, bn_cache = batchnorm_forward(out, self.params['gamma'+str(l+1)], self.params['beta'+str(l+1)], self.bn_params[l])
				
				# relu
				out, relu_cache = relu_forward(out)

				cache = (drop_cache, fc_cache, bn_cache, relu_cache)
				cache_hist.append(cache)
			else:	
				out = np.dot(out, self.params['W'+str(l+1)]) + self.params['b'+str(l+1)]
			out_hist.append(out)

		if mode == 'test':
			probs = np.exp(out) / np.sum(np.exp(out), 1).reshape(-1,1)
			return probs

		# loss
		loss, d_softmax = softmax_loss(out_hist[-1], y)
		for l in range(self.num_layers):
			loss += self.reg * np.sum(self.params['W'+str(l+1)] ** 2)

		# gradiant
		grads = {}
		for k, v in self.params.items():
			grads[k] = np.zeros(v.shape)

		# backward
		for l in range(self.num_layers-1, -1, -1):
			if l == self.num_layers - 1:
				# softmax backward
				grads['b'+str(l+1)] = np.sum(d_softmax, 0)
				grads['W'+str(l+1)] = np.dot(out_hist[l-1].T, d_softmax)
				dx = np.dot(d_softmax, self.params['W'+str(l+1)].T)
			else:
				drop_cache, fc_cache, bn_cache, relu_cache = cache_hist[l]
				
				# relu backward
				dx = relu_backward(dx, relu_cache)
				
				# batch normalize backward
				if self.use_batchnorm:
					dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
					grads['gamma'+str(l+1)] = dgamma
					grads['beta'+str(l+1)] = dbeta
				
				# affine backward
				dx, dw, db = affine_backward(dx, fc_cache)

				# dropout backward
				if self.use_dropout:
					dw = dropout_backward(dw, drop_cache_hist[l])

				grads['b'+str(l+1)] = db
				grads['W'+str(l+1)] = dw
			
			grads['W'+str(l+1)] += 2 * self.reg * self.params['W'+str(l+1)]

		return loss, grads



if __name__ == '__main__':
	main()