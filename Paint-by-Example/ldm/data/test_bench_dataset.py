from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import clip
import bezier

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class COCOImageDataset(data.Dataset):
    def __init__(self,test_bench_dir, id_path):

        self.test_bench_dir=test_bench_dir
        with open(id_path, "r") as f:
            self.id_list = f.readlines()
        self.id_list = [f.strip() for f in self.id_list]
        # self.id_list=np.load(id_path)
        # self.id_list=self.id_list.tolist()
        print("length of test bench",len(self.id_list))
        self.length=len(self.id_list)


    def __getitem__(self, index):
        img_path=os.path.join(os.path.join(self.test_bench_dir,'images', self.id_list[index]+'.jpg'))
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'ref',self.id_list[index] +'.jpg'))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_img=get_tensor_clip()(ref_img)
        ref_image_tensor = ref_img.unsqueeze(0)


        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size


        ### bbox mask
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'masks',self.id_list[index]+'.png'))
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)



        inpaint_tensor=image_tensor*mask_tensor

        return image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},self.id_list[index]



    def __len__(self):
        return self.length



