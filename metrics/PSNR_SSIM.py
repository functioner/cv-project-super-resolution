import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.transform import resize
import os
import matplotlib.pyplot as plt

# Define paths to the folders containing the images
hr_folder = "./hr/"
bicubic_folder = "./bicubic/"
lapsrn_folder = "./lapsrn/"
sparse_folder = "./sparse/"

# Get a list of filenames in the HR folder (assuming all folders have the same file names)
filenames = os.listdir(hr_folder)

# Initialize lists to store metrics
psnr_values_bicubic = []
psnr_values_lapsrn = []
psnr_values_sparse = []
ssim_values_bicubic = []
ssim_values_lapsrn = []
ssim_values_sparse = []
rmse_values_bicubic = []
rmse_values_lapsrn = []
rmse_values_sparse = []

image_names = []  # To store image names

# Loop through each image
for filename in filenames:
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_name = filename.split('_hr')[0]

        original_hr = cv2.imread(os.path.join(hr_folder, filename))
        bicubic = cv2.imread(os.path.join(bicubic_folder, f"{img_name}_bicubic.png"))
        lapsrn = cv2.imread(os.path.join(lapsrn_folder, f"{img_name}_lapsrn.png"))
        sparse = cv2.imread(os.path.join(sparse_folder, f"{img_name}_sparse.png"))

        # Compute PSNR, SSIM, and RMSE values
        psnr_value_bicubic = psnr(original_hr, bicubic)
        psnr_values_bicubic.append(psnr_value_bicubic)
        ssim_value_bicubic = ssim(original_hr, bicubic, win_size=3, multichannel=True)
        ssim_values_bicubic.append(ssim_value_bicubic)
        rmse_value_bicubic = np.sqrt(mse(original_hr, bicubic))
        rmse_values_bicubic.append(rmse_value_bicubic)

        resized_original_hr = resize(original_hr, lapsrn.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
        psnr_value_lapsrn = psnr(resized_original_hr, lapsrn)
        psnr_values_lapsrn.append(psnr_value_lapsrn)
        ssim_value_lapsrn = ssim(resized_original_hr, lapsrn, win_size=3, multichannel=True)
        ssim_values_lapsrn.append(ssim_value_lapsrn)
        rmse_value_lapsrn = np.sqrt(mse(resized_original_hr, lapsrn))
        rmse_values_lapsrn.append(rmse_value_lapsrn)

        psnr_value_sparse = psnr(original_hr, sparse)
        psnr_values_sparse.append(psnr_value_sparse)
        ssim_value_sparse = ssim(original_hr, sparse, win_size=3, multichannel=True)
        ssim_values_sparse.append(ssim_value_sparse)
        rmse_value_sparse = np.sqrt(mse(original_hr, sparse))
        rmse_values_sparse.append(rmse_value_sparse)

        # Save image names
        image_names.append(img_name)

        # Print or store the values as needed
        print(f"{img_name}:")
        print(f"Bicubic PSNR: {psnr_value_bicubic:.2f}")
        print(f"LapSRN PSNR: {psnr_value_lapsrn:.2f}")
        print(f"Sparse PSNR: {psnr_value_sparse:.2f}")
        print(f"Bicubic SSIM: {ssim_value_bicubic:.4f}")
        print(f"LapSRN SSIM: {ssim_value_lapsrn:.4f}")
        print(f"Sparse SSIM: {ssim_value_sparse:.4f}")
        print(f"Bicubic RMSE: {rmse_value_bicubic:.2f}")
        print(f"LapSRN RMSE: {rmse_value_lapsrn:.2f}")
        print(f"Sparse RMSE: {rmse_value_sparse:.2f}")
        print("------------")

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for ax in axs:
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

axs[0].plot(image_names, psnr_values_bicubic, label='Bicubic')
axs[0].plot(image_names, psnr_values_lapsrn, label='LapSRN')
axs[0].plot(image_names, psnr_values_sparse, label='Sparse')
axs[0].set_title('PSNR Comparison')
axs[0].set_xlabel('Image Name')
axs[0].set_ylabel('PSNR')
axs[0].legend(loc='lower right')

axs[1].plot(image_names, ssim_values_bicubic, label='Bicubic')
axs[1].plot(image_names, ssim_values_lapsrn, label='LapSRN')
axs[1].plot(image_names, ssim_values_sparse, label='Sparse')
axs[1].set_title('SSIM Comparison')
axs[1].set_xlabel('Image Name')
axs[1].set_ylabel('SSIM')
axs[1].legend()

axs[2].plot(image_names, rmse_values_bicubic, label='Bicubic')
axs[2].plot(image_names, rmse_values_lapsrn, label='LapSRN')
axs[2].plot(image_names, rmse_values_sparse, label='Sparse')
axs[2].set_title('RMSE Comparison')
axs[2].set_xlabel('Image Name')
axs[2].set_ylabel('RMSE')
axs[2].legend()

plt.tight_layout()
plt.show()
