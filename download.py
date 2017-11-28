import os  
import wget
import config as cfg


args = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

if not os.path.exists(cfg.CIFAR10_PATH):
	os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz') 

'''
cifar-10:
	wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	tar -xzvf cifar-10-python.tar.gz
	rm cifar-10-python.tar.gz 

COCO:
	get imagenet_val:
		wget http://cs231n.stanford.edu/imagenet_val_25.npz

	get coco captioning:
		wget "http://cs231n.stanford.edu/coco_captioning.zip"
		unzip coco_captioning.zip
		rm coco_captioning.zip

	get squezenet_tf
		wget "http://cs231n.stanford.edu/squeezenet_tf.zip"
		unzip squeezenet_tf.zip
		rm squeezenet_tf.zip
'''