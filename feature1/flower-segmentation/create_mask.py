import os
import cv2
import numpy as np
from tqdm import tqdm

data_dir = "./dataset_small/jpg"
seg_dir = "./dataset_small/segmim/"
mask_dir = "./dataset_small/mask/"

file_list = os.listdir(data_dir)
file_list = [file for file in file_list if ".jpg" in file]

for file in tqdm(file_list):
    seg = cv2.imread(os.path.join(seg_dir, file.replace("image", "segmim")))
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 245, 250])
    upper_blue = np.array([130, 255, 260])

    # preparing the mask to overlay
    mask = cv2.inRange(seg, lower_blue, upper_blue)
    mask = 255 - mask
    mask = cv2.erode(mask, np.ones((9, 9)))
    mask = cv2.dilate(mask, np.ones((9, 9)))
    # mask = cv2.GaussianBlur(mask, (9, 9), 1)
    cv2.imwrite(os.path.join(mask_dir, file.replace(".jpg", ".png")), mask)
