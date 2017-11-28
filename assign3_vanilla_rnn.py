from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import time, os, json

import config as cfg
from utils.rnn_layers import *
from utils.image_utils import *
from utils.gradient_check import *
from utils.captioning_solver import *



def main():

	data = cfg.load_coco_data(pca_features=True)

	


	N, D, H = 3, 10, 4

	x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
	prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
	Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
	Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
	b = np.linspace(-0.2, 0.4, num=H)

	next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
	expected_next_h = np.asarray([
	  [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
	  [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
	  [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

	print('next_h error: ', rel_error(expected_next_h, next_h))



if __name__ == '__main__':
	main()







