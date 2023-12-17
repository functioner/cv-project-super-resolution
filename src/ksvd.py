import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning

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
    # Read the input image
    low_input_image_path = 'experiment/fruit/val_lr/pepper.png'
    high_input_image_path = 'data/fruit/val_hr/pepper.png'
    original_image_path =
    original_image = cv2.imread(input_image_path)

    # Ensure the image is in RGB format
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Choose parameters
    scale_factor = 3
    patch_size = 5
    dictionary_size = 256

    # Perform sparse coding super-resolution for RGB image
    super_res_image = super_resolve_ksvd_rgb(original_image, scale_factor, patch_size, dictionary_size)

    # Display the original and super-resolved images
    cv2.imwrite('sample.png', super_res_image)
    # cv2.imshow('Original Image', original_image)
    # cv2.imshow('Sparse Coding Super-Resolved Image', super_res_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

