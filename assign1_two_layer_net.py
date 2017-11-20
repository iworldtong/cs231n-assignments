import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import utils.gradient_check as grad_ck
import utils.vis_utils as vis_utils
from tensorflow.examples.tutorials.mnist import input_data



def show_net_weights(net, data_set):
	plt.figure("W1")
	W1 = net.params['W1']
	if data_set == "cifar10":
		W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
		plt.imshow(vis_utils.visualize_grid(W1, padding=3).astype('uint8'))
	elif data_set == "mnist":
		W1 = np.tile(W1, (3,1))
		W1 = W1.reshape(28, 28, 3, -1).transpose(3, 0, 1, 2)
		
		plt.imshow(vis_utils.visualize_grid(W1, padding=3).astype('uint8'))
	plt.gca().axis('off')


def main(data_set="mnist"):

	if data_set == "cifar10":
		# load cifar10 data set
		train_images, train_labels, test_images, test_labels = cfg.load_cifar10()
		train_images = train_images.astype(float)
		test_images = test_images.astype(float)
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

	num_classes = len(classes)
	# Split the data into train, train_dev, val, and test sets
	num_train = 49000
	num_val   = 1000
	num_test  = 1000
	num_dev   = 100

	# get val set
	mask = list(range(num_train, num_train + num_val))
	val_images = train_images[mask]
	val_labels = train_labels[mask]

	# make a development set, which is a small subset of the training set
	mask = np.random.choice(num_train, num_dev, replace=False)
	dev_images = train_images[mask]
	dev_labels = train_labels[mask]

	# get test set
	mask = list(range(num_test))
	test_images = test_images[mask]
	test_labels = test_labels[mask]
	
	#---------------two layer network---------------#
	input_size = train_images.shape[1]
	hidden_size = 50
	
	np.random.seed(0)

	net = two_layer_net(input_size, hidden_size, num_classes)
	#net.gradiant_check(train_images, train_labels) # !!! will use too much time !!!

	stats = net.train(train_images, train_labels, num_iters=20)

	print('Final training loss: ', stats['loss_history'][-1])
	print('Final training acc: ', stats['train_acc_history'][-1])
	
	# plot the loss history
	plt.subplot(2, 1, 1)
	plt.plot(stats['loss_history'])
	plt.title('Loss history')
	plt.ylabel('Loss')

	plt.subplot(2, 1, 2)
	plt.plot(stats['train_acc_history'], label='train')
	plt.title('Classification accuracy history')
	plt.xlabel('Batch Epoch')
	plt.ylabel('Clasification accuracy')

	# visualize the weights of the network
	show_net_weights(net, data_set)

	plt.show()


class two_layer_net(object):
	def __init__(self, input_size, hidden_size, output_size):
		self.params = {}
		u1 = 2.0 * np.sqrt(6.0 / (input_size + hidden_size))
		self.params['W1'] = (np.random.rand(input_size, hidden_size) - 0.5) * u1
		self.params['b1'] = np.zeros(hidden_size, dtype=float)
		u2 = 2.0 * np.sqrt(6.0 / (hidden_size + output_size))
		self.params['W2'] = (np.random.rand(hidden_size, output_size) - 0.5) * u2
		self.params['b2'] = np.zeros(output_size, dtype=float)

		self.reg=5e-5
		self.lr = 1e-1
		self.lr_decay = 1
		self.show_step = 100
		self.batch_size = 200

	def train(self, train_data, train_labels, num_iters=10, disp=True):
		stats={}
		stats['loss_history'] = []
		stats['train_acc_history'] = []
		loss, grads = self.forward(train_data, train_labels, reg=self.reg)
		acc = self.evaluate(train_data, train_labels)
		print("Random initialization loss :", loss)
		print("Random initialization acc :", acc)
		
		num_data = train_data.shape[0]
		iters = 1
		step = 1
		while iters <= num_iters:
			cnt = 0
			while cnt < num_data:
				for param_name in grads:
					self.params[param_name] = self.params[param_name] - self.lr * np.power(self.lr_decay, step-1) * grads[param_name] - self.reg * self.params[param_name]
				x = train_data[cnt : (cnt + self.batch_size)]
				y = train_labels[cnt : (cnt + self.batch_size)]
				loss, grads = self.forward(x, y, reg=self.reg) 
				stats['loss_history'].append(loss)

				acc = self.evaluate(x, y)
				stats['train_acc_history'].append(acc)

				cnt += self.batch_size
				step += 1
				if disp and step % self.show_step == 0:
					print('Training  ', 'Iter :', iters, '/', num_iters, \
										'  Step :', step, \
										'  Loss :', loss, \
										'  Acc :', acc  
						)
			iters += 1

		return stats

	def forward(self, input_data, labels=None, reg=0.0):
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, D = input_data.shape

		hidden1 = np.maximum(0, np.dot(input_data, W1) + b1)
		hidden2 = np.dot(hidden1, W2) + b2
		neg_hidden2 = hidden2 - np.max(hidden2, axis=1, keepdims=True)
		exp_hidden2 = np.exp(neg_hidden2)
		sum_exp_hidden2 = np.sum(exp_hidden2, axis=1,  keepdims=True)
		scores = exp_hidden2 / sum_exp_hidden2

		# normal forward
		if labels is None:
			return scores

		# calc softmax loss
		loss = - np.sum(neg_hidden2[list(range(len(labels))), labels]) + np.sum(np.log(sum_exp_hidden2))
		loss = loss / N + 0.5 * reg * (np.sum(W1**2) + np.sum(W1**2))

		# calc gradiant
		grads = {}
		grads['W1'] = np.zeros(W1.shape)
		grads['b1'] = np.zeros(b1.shape)
		grads['W2'] = np.zeros(W2.shape)
		grads['b2'] = np.zeros(b2.shape)

		# 梯度公式详见博客
		sgn_mat_y_i = np.zeros(scores.shape)
		sgn_mat_y_i[list(range(len(labels))), labels] = 1
		dz2 = scores - sgn_mat_y_i
		dz1 = np.dot(dz2, W2.T)
		dz1[hidden1 <= 0.0] = 0.0 

		grads['W2'] = np.dot(hidden1.T, dz2) / N + reg * W2
		grads['b2'] = np.mean(dz2, axis=0)
		grads['W1'] = np.dot(input_data.T, dz1) / N + reg * W1
		grads['b1'] = np.mean(dz1, axis=0)

		return loss, grads

	def evaluate(self, input_data, labels):
		preds = self.predict(input_data)
		return np.mean(preds == labels)

	def predict(self, test_data):
		predict_labels = np.argmax(self.forward(test_data), axis=1)		
		return predict_labels

	def gradiant_check(self, X, y, reg=0.05):
		# graidant check
		# If the implementation is correct, the difference between the numeric and
		# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.
		print("Check gradients:")
		loss, grads = self.forward(X, y, reg=reg)
		for param_name in grads:
			f = lambda W: self.forward(X, y, reg=reg)[0]
			param_grad_num = grad_ck.eval_numerical_gradient(f, self.params[param_name], verbose=False)
			print('%s max relative error: %e' % (param_name, grad_ck.rel_error(param_grad_num, grads[param_name])))
		print('\n')



if __name__ == '__main__':
	main(data_set="mnist")