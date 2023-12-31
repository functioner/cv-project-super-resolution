import numpy as np 
from skimage.transform import rescale
from skimage.io import imread, imsave
from os import listdir
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--val_hr', help='path to validation high resolution')
parser.add_argument('--val_lr', help='path to validation low resolution')

args = parser.parse_args()

# Set train and val HR and LR paths
# train_hr_path = 'data/train_hr/'
# train_lr_path = 'data/train_lr/'
# val_hr_path = 'data/val_hr/'
# val_lr_path = 'data/val_lr/'
val_hr_path = args.val_hr + '/'
val_lr_path = args.val_lr + '/'

#numberOfImagesTrainHR = len(listdir(train_hr_path))
#
#for i in tqdm(range(numberOfImagesTrainHR)):
#    img_name = listdir(train_hr_path)[i]
#    img = imread('{}{}'.format(train_hr_path, img_name))
#    new_img = rescale(img, (1/3), anti_aliasing=1)
#    imsave('{}{}'.format(train_lr_path, img_name), new_img, quality=100)

numberOfImagesValHR = len(listdir(val_hr_path))

for i in tqdm(range(numberOfImagesValHR)):
    img_name = listdir(val_hr_path)[i]
    img = imread('{}{}'.format(val_hr_path, img_name))
    # new_img = rescale(img, (1/3), anti_aliasing=1)
    new_img = rescale(img, 1.0/3, anti_aliasing=True, multichannel=True)
    # imsave('{}{}'.format(val_lr_path, img_name), new_img, quality=100)
    imsave('{}{}'.format(val_lr_path, img_name), new_img)
