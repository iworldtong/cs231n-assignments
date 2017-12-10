import os, json
import pickle
import numpy as np
import h5py



SRC_PATH = "./src"
MODEL_PATH = "./model"
DATA_PATH = "E:/data"
MNIST_PATH = os.path.join(DATA_PATH, "mnist")
CIFAR10_PATH = os.path.join(DATA_PATH, "cifar-10-batches-py")

CIFAR10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
				   'dog', 'frog', 'horse', 'ship', 'truck' ]

COCO_BASE_DIR = os.path.join(DATA_PATH, "coco_captioning") 



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



def load_coco_data(base_dir=COCO_BASE_DIR,
                   max_train=None,
                   pca_features=True):
    data = {}
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls



if __name__ == '__main__':
	train_images, train_labels, val_images, val_labels, test_images, test_labels = load_cifar10()

	print('cifar-10 : ')
	print('Training data shape: ', train_images.shape)
	print('Training labels shape: ', train_labels.shape)
	print('Validation data shape: ', val_images.shape)
	print('Validation labels shape: ', val_labels.shape)
	print('Test data shape: ', test_images.shape)
	print('Test labels shape: ', test_labels.shape)


	data = load_coco_data(pca_features=True)
	print('COCO : ')
	for k, v in data.items(): 
		if type(v) == np.ndarray:
			print(k, type(v), v.shape, v.dtype)
		else:
			print(k, type(v), len(v))