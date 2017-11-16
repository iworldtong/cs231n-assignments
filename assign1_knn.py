import config as cfg

import numpy as np
import matplotlib.pyplot as plt



def show_example(train_images, train_labels, classes, num_classes, samples_per_class = 7):
	# Visualize some examples from the dataset.
	# We show a few examples of training images from each class.
	for y, class_name in enumerate(classes):
		idxs = np.flatnonzero(train_labels == y)
		idxs = np.random.choice(idxs, samples_per_class, replace=False)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(samples_per_class, num_classes, plt_idx)
			plt.imshow(train_images[idx].reshape(3, 32, 32).transpose(1,2,0))
			plt.axis('off')
			if i == 0: plt.title(class_name)
	plt.show()


def main():

	k_list = [1, 3, 6]

	# load cifar10 data set
	train_images, train_labels, test_images, test_labels = cfg.load_cifar10()
	classes = cfg.CIFAR10_classes
	num_classes = len(classes)

	# display some examples
	#show_example(train_images, train_labels, classes, num_classes)

	# Subsample the data for more efficient code execution in this exercise
	num_train = 5000
	mask = list(range(num_train))
	train_images = train_images[mask]
	train_labels = train_labels[mask]
	num_test = 500
	mask = list(range(num_test))
	test_images = test_images[mask]
	test_labels = test_labels[mask]

	# Build k nearest neighbor classifier
	classifier = knn()
	
	for k in k_list:
		# training
		classifier.train(train_images, train_labels)

		# predict
		predict_labels = classifier.predict(test_images, k=k)

		# calc acc
		acc = np.sum(np.array(predict_labels == test_labels).astype(float)) / num_test
		print("k :", k,"  Test accuracy :", acc)

	
class knn(object):
	def __init__(self):
		pass

	def train(self, input_data, labels):
		self.train_data   = input_data
		self.train_labels = labels
		
	def predict(self, test_data, k=1):
		num_test = test_data.shape[0]
		predict_labels = np.zeros(num_test)
		dists = self.calc_distances(test_data)
		for i in range(num_test):
			# find the k nearest neighbors of testing points
			k_closest_labels = self.train_labels[np.argsort(dists[i])[0:k]]
			# find the most common label in k_closest_labels
			predict_labels[i] = np.bincount(k_closest_labels).argmax()
		return predict_labels

	def calc_distances(self, test_data):
		'''
			calc L2 distance with no loop
		'''
		num_test = test_data.shape[0]
		num_train = self.train_data.shape[0]
		dists = np.zeros((num_test, num_train))
		dists += np.sum(test_data**2, axis=1).reshape(num_test, 1)
		dists += np.sum(self.train_data**2, axis=1).reshape(1, num_train)
		dists -= 2 * np.dot(test_data, self.train_data.T)
		return dists

if __name__ == '__main__':
	main()

