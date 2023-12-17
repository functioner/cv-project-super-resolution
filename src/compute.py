import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from skimage.color import rgb2ycbcr, ycbcr2rgb
from sklearn.metrics import mean_squared_error
import sys

def super_resolve_ksvd_rgb(image, scale_factor, patch_size, dictionary_size):
    # Extract low-resolution patches
    low_res_patches = []
    height, width, _ = image.shape
    for y in range(0, height - patch_size + 1, scale_factor):
        for x in range(0, width - patch_size + 1, scale_factor):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            low_res_patches.append(patch.flatten())

    low_res_patches = np.array(low_res_patches)

    # Learn a dictionary from the low-resolution patches
    dico = DictionaryLearning(n_components=dictionary_size, max_iter=50)
    dico.fit(low_res_patches[np.random.choice(low_res_patches.shape[0], size=20, replace=False), :])

    upscaled_image = np.zeros((height * scale_factor, width * scale_factor, 3), dtype=np.float32)
    count = np.zeros((height * scale_factor, width * scale_factor, 3), dtype=np.float32)

    idx = 0
    for y in range(0, height - patch_size + 1, scale_factor):
        for x in range(0, width - patch_size + 1, scale_factor):
            patch = dico.components_.T @ dico.transform(image[y:y+patch_size, x:x+patch_size, :].flatten())
            patch = patch.reshape((patch_size*scale_factor, patch_size*scale_factor, 3))
            upscaled_image[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor, :] += patch
            count[y:y+patch_size*scale_factor, x:x+patch_size*scale_factor, :] += 1
            idx += 1

    upscaled_image /= count
    upscaled_image = np.clip(upscaled_image, 0, 255).astype(np.uint8)

    return upscaled_image

def main():
    a = sys.argv[1]
    b = sys.argv[2]
    a = cv2.imread(a)
    b = cv2.imread(b)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[0], a.shape[1]))

    # Ensure the image is in RGB format
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    c = 0
    d = 0
    H, W, _ = a.shape
    for i in range(H):
        for j in range(W):
            for k in range(3):
                if a[i,j,k] != b[i,j,k]:
                    c += 1
                else:
                    d += 1
    print('diff =', c, '    same =', d)

    a = rgb2ycbcr(a)[:, :, 0]
    b = rgb2ycbcr(b)[:, :, 0]
    print(a.shape, b.shape)
    e = np.sqrt(mean_squared_error(a, b))
    e = np.zeros((1,)) + e
    print('rmse =', e)

    # Display the original and super-resolved images

if __name__ == "__main__":
    main()

