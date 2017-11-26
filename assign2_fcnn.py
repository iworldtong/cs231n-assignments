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

	small_data = {
	  'X_train': train_images[:1000],
	  'y_train': train_labels[:1000],
	  'X_val': val_images,
	  'y_val': val_labels,
	}

	data = {
	  'X_train': train_images,
	  'y_train': train_labels,
	  'X_val': val_images,
	  'y_val': val_labels,
	}

	dropout_cmp(small_data)
	
	#update_cmp(data)



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
		drop_cache_hist = []
		# forward
		out = X.copy()
		for l in range(self.num_layers):
			if l < (self.num_layers - 1):		
				# dropout
				if self.use_dropout:
					temp_W, drop_cache = dropout_forward(self.params['W'+str(l+1)], self.dropout_param)
					drop_cache_hist.append(drop_cache)
				else:
					temp_W = self.params['W'+str(l+1)]

				out, cache = affine_relu_forward(out, temp_W, self.params['b'+str(l+1)])
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
				dx, dw, db = affine_relu_backward(dx, cache_hist[l])
				if self.use_dropout:
					dw = dropout_backward(dw, drop_cache_hist[l])
				grads['b'+str(l+1)] = db
				grads['W'+str(l+1)] = dw
			
			grads['W'+str(l+1)] += 2 * self.reg * self.params['W'+str(l+1)]

		return loss, grads

def dropout_cmp(data):
	solvers = {}
	dropout_choices = [0, 0.25, 0.5, 0.75]
	for dropout in dropout_choices:
	  model = FullyConnectedNet([500], dropout=dropout)
	  print(dropout)

	  solver = Solver(model, data,
	                  num_epochs=25, batch_size=100,
	                  update_rule='adam',
	                  optim_config={
	                    'learning_rate': 5e-4,
	                  },
	                  verbose=True, print_every=100)
	  solver.train()
	  solvers[dropout] = solver

	# Plot train and validation accuracies of the two models
	train_accs = []
	val_accs = []
	for dropout in dropout_choices:
	  solver = solvers[dropout]
	  train_accs.append(solver.train_acc_history[-1])
	  val_accs.append(solver.val_acc_history[-1])

	plt.subplot(2, 1, 1)
	for dropout in dropout_choices:
	  plt.plot(solvers[dropout].train_acc_history, '-o', label='%.2f dropout' % dropout)
	plt.title('Train accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(ncol=2, loc='lower right')
	  
	plt.subplot(2, 1, 2)
	for dropout in dropout_choices:
	  plt.plot(solvers[dropout].val_acc_history, '-o', label='%.2f dropout' % dropout)
	plt.title('Val accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(ncol=2, loc='lower right')

	plt.gcf().set_size_inches(15, 15)
	plt.show()

def update_cmp(data):
	solvers = {}
	learning_rates = {'sgd':1e-1, 'sgd_momentum':1e-1, 'rmsprop': 1e-4, 'adam': 1e-3}
	update_rule_list = ['sgd', 'sgd_momentum', 'rmsprop', 'adam']
	for update_rule in update_rule_list:
		print('running with ', update_rule)
		model = FullyConnectedNet([100, 50], \
								  reg=5e-5, \
								  weight_scale=5e-2)

		solver = Solver(model, data, \
	                  	num_epochs=20, batch_size=200, \
	                  	update_rule=update_rule, \
	                  	optim_config={
	                   	 'learning_rate': learning_rates[update_rule]
	                  	}, \
	                  	verbose=True)
		solvers[update_rule] = solver
		solver.train()
		print()

	plt.subplot(3, 1, 1)
	plt.title('Training loss')
	plt.xlabel('Iteration')

	plt.subplot(3, 1, 2)
	plt.title('Training accuracy')
	plt.xlabel('Epoch')

	plt.subplot(3, 1, 3)
	plt.title('Validation accuracy')
	plt.xlabel('Epoch')

	for update_rule, solver in list(solvers.items()):
		plt.subplot(3, 1, 1)
		plt.plot(solver.loss_history, 'o', label=update_rule)
	  
		plt.subplot(3, 1, 2)
		plt.plot(solver.train_acc_history, '-o', label=update_rule)

		plt.subplot(3, 1, 3)
		plt.plot(solver.val_acc_history, '-o', label=update_rule)
	  
	for i in range(1, 4):
		plt.subplot(3, 1, i)
		plt.legend(loc='upper center', ncol=4)
	plt.gcf().set_size_inches(15, 15)
	plt.show()

if __name__ == '__main__':
	main()