import sys
import cv2
import numpy as np
from math import log10, sqrt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                   # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min()) * 255.0

def compute_rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

def compute_psnr(img1, img2):
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    return peak_signal_noise_ratio(img1_norm, img2_norm)

def compute_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True, data_range=255)

def main():
    # Check if two file paths are provided as command line arguments
    if len(sys.argv) != 3:
        print("Usage: python super_resolution_metrics.py <image1_path> <image2_path>")
        sys.exit(1)

    # Read images
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Check if images are loaded successfully
    if img1 is None or img2 is None:
        print("Error: Couldn't read one or both images.")
        sys.exit(1)

    # Ensure images have the same shape
    if img1.shape != img2.shape:
        print("Error: Images must have the same dimensions.")
        sys.exit(1)

    # Convert images to floating point for metric calculations
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Compute metrics
    rmse_value = compute_rmse(img1, img2)
    #psnr_value = compute_psnr(img1, img2)
    psnr_value = PSNR(img1, img2)
    ssim_value = compute_ssim(img1, img2)

    # Print results
    print(f"RMSE: {rmse_value:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    main()

