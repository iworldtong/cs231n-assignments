from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import time, os, json

import config as cfg
from utils.layers import *
from utils.rnn_layers import *
from utils.image_utils import *
from utils.gradient_check import *
from utils.captioning_solver import *

def overfit_traning_set():

	np.random.seed(231)

	small_data = cfg.load_coco_data(max_train=50)

	small_lstm_model = rnn(
	          cell_type='lstm',
	          word_to_idx=small_data['word_to_idx'],
	          input_dim=small_data['train_features'].shape[1],
	          hidden_dim=512,
	          wordvec_dim=256,
	          dtype=np.float32,
	        )

	small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,
	           update_rule='adam',
	           num_epochs=50,
	           batch_size=25,
	           optim_config={
	             'learning_rate': 5e-3,
	           },
	           lr_decay=0.995,
	           verbose=True, print_every=10,
	         )

	small_lstm_solver.train()


	for split in ['train', 'val']:
		minibatch = cfg.sample_coco_minibatch(small_data, split=split, batch_size=2)
		gt_captions, features, urls = minibatch
		gt_captions = cfg.decode_captions(gt_captions, small_data['idx_to_word'])

		sample_captions = small_lstm_model.sample(features, max_length=15)
		sample_captions = cfg.decode_captions(sample_captions, small_data['idx_to_word'])

		for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
			plt.imshow(image_from_url(url))
			plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
			plt.axis('off')
			plt.show()



def main():

	data = cfg.load_coco_data()

	overfit_traning_set()
	




class rnn(object):
	"""
	A CaptioningRNN produces captions from image features using a recurrent
	neural network.

	The RNN receives input vectors of size D, has a vocab size of V, works on
	sequences of length T, has an RNN hidden dimension of H, uses word vectors
	of dimension W, and operates on minibatches of size N.

	Note that we don't use any regularization for the CaptioningRNN.
	"""
	def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
				 hidden_dim=128, cell_type='rnn', dtype=np.float32):
		"""
		Construct a new CaptioningRNN instance.

		Inputs:
		- word_to_idx: A dictionary giving the vocabulary. It contains V entries,
		  and maps each string to a unique integer in the range [0, V).
		- input_dim: Dimension D of input image feature vectors.
		- wordvec_dim: Dimension W of word vectors.
		- hidden_dim: Dimension H for the hidden state of the RNN.
		- cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
		- dtype: numpy datatype to use; use float32 for training and float64 for
		  numeric gradient checking.
		"""
		if cell_type not in {'rnn', 'lstm'}:
			raise ValueError('Invalid cell_type "%s"' % cell_type)

		self.cell_type = cell_type
		self.dtype = dtype
		self.word_to_idx = word_to_idx
		self.idx_to_word = {i: w for w, i in word_to_idx.items()}
		self.params = {}

		vocab_size = len(word_to_idx)

		self._null = word_to_idx['<NULL>']
		self._start = word_to_idx.get('<START>', None)
		self._end = word_to_idx.get('<END>', None)

		# Initialize word vectors
		self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
		self.params['W_embed'] /= 100

		# Initialize CNN -> hidden state projection parameters
		self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
		self.params['W_proj'] /= np.sqrt(input_dim)
		self.params['b_proj'] = np.zeros(hidden_dim)

		# Initialize parameters for the RNN
		dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
		self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
		self.params['Wx'] /= np.sqrt(wordvec_dim)
		self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
		self.params['Wh'] /= np.sqrt(hidden_dim)
		self.params['b'] = np.zeros(dim_mul * hidden_dim)

		# Initialize output to vocab weights
		self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
		self.params['W_vocab'] /= np.sqrt(hidden_dim)
		self.params['b_vocab'] = np.zeros(vocab_size)

		# Cast parameters to correct dtype
		for k, v in self.params.items():
			self.params[k] = v.astype(self.dtype)


	def loss(self, features, captions):
		"""
		Compute training-time loss for the RNN. We input image features and
		ground-truth captions for those images, and use an RNN (or LSTM) to compute
		loss and gradients on all parameters.

		Inputs:
		- features: Input image features, of shape (N, D)
		- captions: Ground-truth captions; an integer array of shape (N, T) where
		  each element is in the range 0 <= y[i, t] < V

		Returns a tuple of:
		- loss: Scalar loss
		- grads: Dictionary of gradients parallel to self.params
		"""
		# Cut captions into two pieces: captions_in has everything but the last word
		# and will be input to the RNN; captions_out has everything but the first
		# word and this is what we will expect the RNN to generate. These are offset
		# by one relative to each other because the RNN should produce word (t+1)
		# after receiving word t. The first element of captions_in will be the START
		# token, and the first element of captions_out will be the first word.
		captions_in = captions[:, :-1]
		captions_out = captions[:, 1:]

		# You'll need this
		mask = (captions_out != self._null)

		# In the forward pass you will need to do the following:                   #
		# (1) Use an affine transformation to compute the initial hidden state     #
		#     from the image features. This should produce an array of shape (N, H)#
		# (2) Use a word embedding layer to transform the words in captions_in     #
		#     from indices to vectors, giving an array of shape (N, T, W).         #
		# (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
		#     process the sequence of input word vectors and produce hidden state  #
		#     vectors for all timesteps, producing an array of shape (N, T, H).    #
		# (4) Use a (temporal) affine transformation to compute scores over the    #
		#     vocabulary at every timestep using the hidden states, giving an      #
		#     array of shape (N, T, V).                                            #
		# (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
		#     the points where the output word is <NULL> using the mask above.     #

		# Weight and bias for the affine transform from image features to initial
		# hidden state
		W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
		h0 = np.dot(features, W_proj) + b_proj

		# Word embedding matrix
		W_embed = self.params['W_embed']
		word_embedding_out, word_embedding_cache = word_embedding_forward(captions_in, W_embed)

		# Input-to-hidden, hidden-to-hidden, and biases for the RNN
		Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
		if self.cell_type == 'rnn':
			rnn_out, rnn_cache = rnn_forward(word_embedding_out, h0, Wx, Wh, b)
		elif self.cell_type == 'lstm':
			rnn_out, rnn_cache = lstm_forward(word_embedding_out, h0, Wx, Wh, b)

		# Weight and bias for the hidden-to-vocab transformation.
		N, T, H = rnn_out.shape
		W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
		out, vocab_cache = temporal_affine_forward(rnn_out, W_vocab, b_vocab)
		
		# In the backward pass you will need to compute the gradient of the loss   #
		# with respect to all model parameters. Use the loss and grads variables   #
		# defined above to store loss and gradients; grads[k] should give the      #
		# gradients for self.params[k].                                            #
		loss, dout = temporal_softmax_loss(out, captions_out, mask, verbose=False)
		
		grads = {}
		dout, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, vocab_cache)
		if self.cell_type == 'rnn':
			dout, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dout, rnn_cache)
		elif self.cell_type == 'lstm':
			dout, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dout, rnn_cache)
		grads['W_embed'] = word_embedding_backward(dout, word_embedding_cache)
		_, grads['W_proj'], grads['b_proj'] = affine_backward(dh0, (features, W_proj, b_proj))

		return loss, grads


	def sample(self, features, max_length=30):
		"""
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
		"""
		N = features.shape[0]
		captions = self._null * np.ones((N, max_length), dtype=np.int32)

		# Unpack parameters
		W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
		W_embed = self.params['W_embed']
		Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
		W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

		###########################################################################
		# TODO: Implement test-time sampling for the model. You will need to      #
		# initialize the hidden state of the RNN by applying the learned affine   #
		# transform to the input image features. The first word that you feed to  #
		# the RNN should be the <START> token; its value is stored in the         #
		# variable self._start. At each timestep you will need to do to:          #
		# (1) Embed the previous word using the learned word embeddings           #
		# (2) Make an RNN step using the previous hidden state and the embedded   #
		#     current word to get the next hidden state.                          #
		# (3) Apply the learned affine transformation to the next hidden state to #
		#     get scores for all words in the vocabulary                          #
		# (4) Select the word with the highest score as the next word, writing it #
		#     to the appropriate slot in the captions variable                    #
		#                                                                         #
		# For simplicity, you do not need to stop generating after an <END> token #
		# is sampled, but you can if you want to.                                 #
		#                                                                         #
		# HINT: You will not be able to use the rnn_forward or lstm_forward       #
		# functions; you'll need to call rnn_step_forward or lstm_step_forward in #
		# a loop.                                                                 #
		###########################################################################
		current_h, _ = affine_forward(features, W_proj, b_proj)
		current_c = np.zeros_like(current_h)
		captions[:, 0] = self._start

		for t in range(1, max_length):
			current_x, _ = word_embedding_forward(captions[:, t], W_embed)
			if self.cell_type == 'rnn':
				current_h, _ = rnn_step_forward(current_x, current_h, Wx, Wh, b)
			elif self.cell_type == 'lstm':
				current_h, current_c, _ = lstm_step_forward(current_x, current_h, current_c, Wx, Wh, b)
				
			out, _ = affine_forward(current_h, W_vocab, b_vocab)
			captions[:, t] = np.argmax(out, axis=1)

		return captions
		


if __name__ == '__main__':
	main()







