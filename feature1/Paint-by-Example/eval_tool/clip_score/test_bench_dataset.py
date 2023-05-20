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
    def __init__(self,test_bench_dir,result_dir, id_path):

        self.test_bench_dir=test_bench_dir
        self.result_dir=result_dir
        with open(id_path, "r") as f:
            self.id_list = f.readlines()
        self.id_list = [f.strip() for f in self.id_list]
        # self.id_list=np.load('test_bench/id_list.npy')
        # self.id_list=self.id_list.tolist()
        print("length of test bench",len(self.id_list))
        self.length=len(self.id_list)




    def __getitem__(self, index):
        result_path=os.path.join(os.path.join(self.result_dir,self.id_list[index]+'.png'))
        result_p = Image.open(result_path).convert("RGB")
        result_tensor = get_tensor_clip()(result_p)

        ### Get reference
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'ref',self.id_list[index] +'.jpg'))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_image_tensor=get_tensor_clip()(ref_img)


        ### bbox mask
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'masks',self.id_list[index]+'.png'))
        mask_img=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        # idx0 = np.nonzero(mask_img.ravel()> 10)[0]
        # idxs = [idx0.min(), idx0.max()]
        # out = np.column_stack(np.unravel_index(idxs,mask_img.shape))
        # crop_tensor=result_tensor[:,out[0][0]:out[1][0],out[0][1]:out[1][1]]
        mean_h = np.max(mask_img, axis=1)
        indices = np.argwhere(mean_h > 10)
        upper = np.min(indices)
        lower = np.max(indices)

        mean_v = np.max(mask_img, axis=0)
        indices = np.argwhere(mean_v > 10)
        left = np.min(indices)
        right = np.max(indices)

        crop_tensor = result_tensor[:, upper:lower, left:right]

        # print(crop_tensor.shape, upper, lower, left, right)
        # if crop_tensor.shape[1] == 0 or crop_tensor.shape[0] == 0 or crop_tensor.shape[2] == 0:
        #     print("==============", self.id_list[index])
        crop_tensor=T.Resize([224,224])(crop_tensor)


        return crop_tensor,ref_image_tensor



    def __len__(self):
        return self.length



