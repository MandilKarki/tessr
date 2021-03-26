import multiprocessing
import os
#from resnet import *

num_workers = multiprocessing.cpu_count()

root_dir = '/home/mandil/code/Computer Vision/ILABS/img classi/data_split/train'
model_name = ''
image_size = (226,226)
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
train_split = 0.8
num_epochs = 1
learning_rate = 0.001
checkpoint_dir = ''

batch_size = 8

checkpoint_save_frequency = 2
checkpoint_filename = model_name + 'path.tar'

#to load pretrained imagenet weights
load_pretrained_weights = True

#to specify whether to train the whole network or the last layer only
train_only_last_layer = True

#full path of checkpoint
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
load_checkpoint = False


