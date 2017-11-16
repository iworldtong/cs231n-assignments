import os  
import wget
import config as cfg


args = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

if not os.path.exists(cfg.CIFAR10_PATH):
	os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz') 

'''
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
'''