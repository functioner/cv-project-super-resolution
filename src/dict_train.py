import numpy as np 
from rnd_smp_patch import rnd_smp_patch
from patch_pruning import patch_pruning
from spams import trainDL
import pickle
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', help='path to training dataset')
parser.add_argument('--dict_prefix', help='path prefix to training result (dict)')

args = parser.parse_args()

dict_size   = 2048         # dictionary size
lmbd        = 0.1          # sparsity regularization
patch_size  = 3            # image patch size
nSmp        = 100000       # number of patches to sample
upscale     = 3            # upscaling factor

# Randomly sample image patches
Xh, Xl = rnd_smp_patch(args.dataset + '/', patch_size, nSmp, upscale)

# Prune patches with small variances
Xh, Xl = patch_pruning(Xh, Xl)
Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

print('Start to train...')

# Dictionary learning
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)
print('finished Dh training')
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)
print('finished Dl training')

# Saving dictionaries to files
with open(args.dict_prefix + 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open(args.dict_prefix + 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
