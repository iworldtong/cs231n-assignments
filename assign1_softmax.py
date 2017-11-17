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
	num_dev   = 1

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
	
	# Softmax Classifier
	classifier = softmax()

	#classifier.train(train_images, train_labels)
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


class softmax(object):
	def __init__(self):
		self.show_step = 100
		self.batch_size=128
		self.lr=1e-8 
		self.reg=1e-9

	def train(self, input_data, labels, num_classes, 
				num_iters=10, 
				init_W=True, 
				disp=True):

		num_data = input_data.shape[0]
		loss_list = []

		if init_W:
			self.W = np.random.randn(input_data.shape[1], num_classes) * 0.0001
		loss, grad = self.get_loss_and_grad(self.W, input_data, labels, reg=self.reg)

		iters = 1
		step = 1
		while iters <= num_iters:
			cnt = 0
			while cnt < num_data:
				self.W = self.W - self.lr * grad - self.reg * self.W
				x = input_data[cnt : (cnt + self.batch_size)]
				y = labels[cnt : (cnt + self.batch_size)]
				loss, grad = self.get_loss_and_grad(self.W, x, y, reg=self.reg)
				loss_list.append(loss)

				cnt += self.batch_size
				step += 1
				if disp and step % self.show_step == 0:
					print('Training  ', 'Iter :', iters, '/', num_iters,'  Step :', step, '  Loss :', loss)
			iters += 1

		return loss_list


	def get_loss_and_grad(self, W, test_data, test_labels, reg=0.1):
		num_data = len(test_labels)
		dW = np.zeros(W.shape, dtype=float)
		score_matrix = np.exp(np.dot(test_data, W))
		sum_score_vec = np.sum(score_matrix, axis=1)
		normal_score_matrix = score_matrix / sum_score_vec.reshape(-1,1)

		# calc loss		
		loss_vec = score_matrix[list(range(num_data)), test_labels] / sum_score_vec
		loss_vec = -np.log(loss_vec)
		loss = np.sum(loss_vec) + 0.5 * reg * np.sum(W ** 2)

		# calc gradient	
		for i in range(num_data):
			temp = normal_score_matrix[i] * test_data[i].reshape(-1,1)
			temp[:, test_labels[i]] -= test_data[i]
			dW += temp

		return loss, dW

	def predict(self, input_data):
		score_matrix = np.dot(input_data, self.W)
		predict_labels = np.argmax(score_matrix, axis=1)
		return predict_labels



if __name__ == '__main__':
	main()