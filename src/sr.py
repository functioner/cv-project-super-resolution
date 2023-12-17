import numpy as np 
from os import listdir
from sklearn.decomposition import DictionaryLearning
import cv2
import argparse
import random
from os.path import isfile, join
import logging

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

def read_image(path):
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

def upscale(origin, Dl, Dh, scale_factor, patch_size):
    H, W, _ = origin.shape
    patches = []
    for x in range(H - patch_size + 1):
        for y in range(W - patch_size + 1):
            patches.append(origin[x:x+patch_size, y:y+patch_size, :].flatten())
    patches = np.array(patches)
    coefficients = Dl.transform(patches)

    image = np.zeros((H * scale_factor, W * scale_factor, 3), dtype=np.float32)
    count = np.zeros((H * scale_factor, W * scale_factor, 3), dtype=np.float32)
    idx = 0
    for x in range(H - patch_size + 1):
        for y in range(W - patch_size + 1):
            patch = Dh.components_.T @ coefficients[idx, :]
            image[x*scale_factor:x*scale_factor+patch_size*scale_factor, y*scale_factor:y*scale_factor+patch_size*scale_factor, :] += patch.reshape(patch_size*scale_factor, patch_size*scale_factor, 3)
            count[x*scale_factor:x*scale_factor+patch_size*scale_factor, y*scale_factor:y*scale_factor+patch_size*scale_factor, :] += 1
            idx += 1
    image /= count
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_hr', help='path to training high resolution')
parser.add_argument('--train_lr', help='path to training low resolution')
parser.add_argument('--val_hr', help='path to validation high resolution')
parser.add_argument('--val_lr', help='path to validation low resolution')
parser.add_argument('--result', help='path to super resolution result')

args = parser.parse_args()

scale_factor = 3
patch_size = 5
dictionary_size = 200
samples = 10000

data_lr = []
data_hr = []

logging.info('start to load training data')

for f in listdir(args.train_hr):
    if isfile(join(args.train_lr, f)) and isfile(join(args.train_hr, f)):
        data_lr.append(read_image(join(args.train_lr, f)))
        data_hr.append(read_image(join(args.train_hr, f)))

lr_patches = []
hr_patches = []

logging.info('start to make patches')

for i in np.arange(len(data_lr))[np.random.choice(len(data_hr), size=samples, replace=True)]:
    lr, hr = data_lr[i], data_hr[i]
    H, W, _ = lr.shape
    while True:
        x = random.randint(0, H - patch_size + 1)
        y = random.randint(0, W - patch_size + 1)
        lr_patch = lr[x:x+patch_size, y:y+patch_size, :].flatten()
        hr_patch = hr[x:x+patch_size*scale_factor, y:y+patch_size*scale_factor, :].flatten()
        if lr_patch.shape[0] != patch_size*patch_size*3 or hr_patch.shape[0] != patch_size*patch_size*scale_factor*scale_factor*3:
            continue
        lr_patches.append(lr_patch)
        hr_patches.append(hr_patch)
        break

lr_patches, hr_patches = np.vstack(lr_patches), np.vstack(hr_patches)

logging.info('start to learn low resolution dict')

Dl = DictionaryLearning(n_components=dictionary_size, max_iter=100, verbose=True).fit(lr_patches)

logging.info('start to learn high resolution dict')

Dh = DictionaryLearning(n_components=dictionary_size, max_iter=100, verbose=True).fit(hr_patches)

logging.info('start to reconstruct')

for f in listdir(args.val_lr):
    if isfile(join(args.val_lr, f)):
        origin = read_image(join(args.val_lr, f))
        image = upscale(origin, Dl, Dh, scale_factor, patch_size)
        cv2.imwrite(join(args.result, f), image)