import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import gradient_check as grad_ck


def main():

	# load cifar10 data set
	train_images, train_labels, test_images, test_labels = cfg.load_cifar10()
	train_images = train_images.astype(float)
	test_images = test_images.astype(float)
	classes = cfg.CIFAR10_classes
	num_classes = len(classes)

	# Split the data into train, train_dev, val, and test sets
	num_train = 49000
	num_val   = 1000
	num_test  = 1000
	num_dev   = 500

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

	# zero mean
	mean_train_image = np.mean(train_images, axis=0)
	train_images -= mean_train_image
	val_images -= mean_train_image
	dev_images -= mean_train_image
	test_images -= mean_train_image

	# bias trick : append the bias dimension of ones
	train_images = np.hstack([train_images, np.ones((train_images.shape[0], 1))])
	val_images = np.hstack([val_images, np.ones((val_images.shape[0], 1))])
	dev_images = np.hstack([dev_images, np.ones((dev_images.shape[0], 1))])
	test_images = np.hstack([test_images, np.ones((test_images.shape[0], 1))])
	
	# SVM Classifier
	classifier = linear_svm()

	# training
	loss_hist = classifier.train(train_images, train_labels, num_classes)

	# test
	predict_labels = classifier.predict(test_images)

	# accuracy
	acc = np.mean(predict_labels == test_labels)
	print("Accuracy :", acc)

	# Visualization
	plt.figure("training loss curve")
	plt.plot(loss_hist)
	plt.xlabel('Mini batch step (batch size :'+str(classifier.batch_size) + ')')
	plt.ylabel('Loss value')
	
	# Visualize the learned weights for each class.
	w = classifier.W[:-1,:] # strip out the bias
	w = w.reshape(32, 32, 3, 10)
	w_min, w_max = np.min(w), np.max(w)
	plt.figure("weight matrix visualization")
	for i in range(10):
		plt.subplot(2, 5, i + 1)	      
		# Rescale the weights to be between 0 and 255
		wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
		plt.imshow(wimg.astype('uint8'))
		plt.axis('off')
		plt.title(classes[i])

	plt.show()


class linear_svm(object):
	def __init__(self):
		self.show_step = 100
		self.batch_size=128
		self.lr=1e-8 
		self.reg=1e-9

	def train(self, input_data, labels, num_classes, 
				num_iters=10, 
				init_W=True, 
				disp=True,
				grad_ck_flag=False):

		num_data = input_data.shape[0]
		loss_list = []

		# 初始化权重矩阵，以及代码检查
		if init_W:
			self.W = np.random.randn(input_data.shape[1], num_classes) * 0.0001
			# 当权重很小时， loss 约等于 (num_class - 1) --> 理论值
			loss, grad = self.get_loss_and_grad(self.W, input_data, labels)
			print('Check loss function : initial loss :', "%0.4f"%loss, '  theoretical value :', str(num_classes - 1))

			if grad_ck_flag:
				# 梯度检验 ： 比较公式计算出的梯度与数值梯度的差值，检查代码是否存在bug
				f = lambda W: self.get_loss_and_grad(W, input_data, labels, 0.0)[0]
				print("Check gradient:")
				grad_numerical = grad_ck.grad_check_sparse(f, self.W, grad)

		# 迭代训练
		iters = 1
		step = 1
		while iters <= num_iters:
			cnt = 0
			while cnt < num_data:
				self.W = self.W - self.lr * grad - self.reg * self.W
				x = input_data[cnt : (cnt + self.batch_size)]
				y = labels[cnt : (cnt + self.batch_size)]
				loss, grad = self.get_loss_and_grad(self.W, x, y)
				loss_list.append(loss)

				cnt += self.batch_size
				step += 1
				if disp and step % self.show_step == 0:
					print('Training  ', 'Iter :', iters, '/', num_iters,'  Step :', step, '  Loss :', loss)
			iters += 1

		return loss_list


	def get_loss_and_grad(self, W, test_data, test_labels, reg=0.1, delta=1):
		'''
			calc hinge loss with no loop
		'''
		dW = np.zeros(W.shape, dtype=float)
		num_data = test_data.shape[0]
		loss = 0.0

		# Calc hinge loss with 
		score_matrix = np.dot(test_data, W)
		score_matrix -= score_matrix[list(range(len(test_labels))), test_labels].reshape(-1, 1) 
		score_matrix += delta
		score_matrix[list(range(len(test_labels))), test_labels] = 0 
		score_matrix[score_matrix < 0] = 0
		loss = np.sum(score_matrix) / num_data

		# Add regularization to the loss
		loss += reg * np.sum(W ** 2)

		# calc the gradient of the loss function
		bin_score_matrix = np.zeros(score_matrix.shape)
		bin_score_matrix[score_matrix > 0] = 1
		for i in range(num_data):
			temp = test_data[i].reshape(-1,1) * bin_score_matrix[i]
			temp[:, test_labels[i]] = - np.sum(bin_score_matrix[i]) * test_data[i]
			dW += temp
		dW /= num_data

		return loss, dW

	def predict(self, input_data):
		score_matrix = np.dot(input_data, self.W)
		predict_labels = np.argmax(score_matrix, axis=1)
		return predict_labels

	

if __name__ == '__main__':
	main()