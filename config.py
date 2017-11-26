import os
import pickle
import numpy as np


DATA_PATH = "./data"
MNIST_PATH = os.path.join(DATA_PATH, "mnist")
CIFAR10_PATH = os.path.join(DATA_PATH, "cifar-10-batches-py")

CIFAR10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
				   'dog', 'frog', 'horse', 'ship', 'truck' ]


def unpickle(file):
	with open(file, 'rb') as f:
		dict = pickle.load(f, encoding='bytes')
	return dict

def load_cifar10():
	'''
		vector2image : img = data.reshape(3, 32, 32).transpose(1,2,0)
	'''
	file_list = ['test_batch', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
	test_dict = {}
	train_dict =  {b'data': [], b'labels': []}
	
	for index, file in enumerate(file_list):
		file = os.path.join(CIFAR10_PATH, file)
		if 'test' in file: 
			test_dict = unpickle(file)
		else:
			temp_dict = unpickle(file)
			train_dict[b'data'].extend(temp_dict[b'data'])
			train_dict[b'labels'].extend(temp_dict[b'labels'])

	Train_images, Train_labels, Test_images, Test_labels = np.array(train_dict[b'data']).astype(np.float64) / 255.,\
														   np.array(train_dict[b'labels']).astype(int), \
														   np.array(test_dict[b'data']).astype(np.float64) / 255., \
														   np.array(test_dict[b'labels']).astype(int)
	num_training=49000
	num_validation=1000
	num_test=10000

	# Subsample the data
	mask = range(num_training, num_training + num_validation)
	val_images = Train_images[mask]
	val_labels = Train_labels[mask]
	mask = range(num_training)
	train_images = Train_images[mask]
	train_labels = Train_labels[mask]
	mask = range(num_test)
	test_images = Test_images[mask]
	test_labels = Test_labels[mask]

	# Normalize the data: subtract the mean image
	mean_image = np.mean(Train_images, axis=0)
	train_images -= mean_image
	val_images	 -= mean_image
	test_images  -= mean_image													
	

	return train_images, train_labels, \
		   val_images,   val_labels, \
		   test_images,  test_labels


if __name__ == '__main__':
	train_images, train_labels, val_images, val_labels, test_images, test_labels = load_cifar10()

	print('Training data shape: ', train_images.shape)
	print('Training labels shape: ', train_labels.shape)
	print('Validation data shape: ', val_images.shape)
	print('Validation labels shape: ', val_labels.shape)
	print('Test data shape: ', test_images.shape)
	print('Test labels shape: ', test_labels.shape)