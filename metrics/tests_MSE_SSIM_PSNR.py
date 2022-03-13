import cv2
import numpy as np
from skimage import io
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.color import rgb2gray


path_dir_ref = 'path_reference_images'
path_dir_dist = 'path_undistorted_images/'

def get_paths(path_dir):

    paths_test = glob.glob(path_dir + "*.jpg")
    paths_test.sort()
    paths_test = list(paths_test)

    return paths_test

paths_ref = get_paths(path_dir_ref)
paths_dist = get_paths(path_dir_dist)

mse=[]
ssim_list=[]
PSNR_list = []
  
def PSNR(original, compressed):
    original = np.float64(original) / 255.
    compressed = np.float64(compressed) / 255.
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return "Same Image"
    psnr = 10 * log10(1. / mse)
    return psnr
  
def main():
     original = cv2.imread("original_image.png")
     compressed = cv2.imread("compressed_image.png", 1)
     value = PSNR(original, compressed)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()


for i in range(0, len(paths_ref)):
    img = rgb2gray(io.imread(paths_ref[i]))
    img_dist = rgb2gray(io.imread(paths_dist[i]))

    mse_none = mean_squared_error(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())

    mse_dist = mean_squared_error(img, img_dist)
    ssim_dist = ssim(img, img_dist,
                    data_range=img_dist.max() - img_dist.min())

    mse.append(mse_dist)
    ssim_list.append(ssim_dist)

    original = cv2.imread(paths_ref[i])
    compressed = cv2.imread(paths_dist[i], 1)
    value = PSNR(original, compressed)
    PSNR_list.append(value)
    print(f"PSNR value is {value} dB")

Print('MSE:')
print(sum(mse)/len(mse))
Print('SSIM:')
print(sum(ssim_list)/len(ssim_list))
Print('PSNR:')
print(sum(PSNR_list)/len(PSNR_list))