import sys
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import clip
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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

def read_image(path):
    img = Image.open(path).convert("RGB")
    img = get_tensor_clip()(img)
    img=T.Resize([224,224])(img)
    return img.unsqueeze(0)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default="cuda",
                    help='Device to use. Like cuda, cuda:0 or cpu')

opt = parser.parse_args()
args={}
clip_model,preprocess = clip.load("ViT-B/32", device="cuda")
data_dir = "./dataset/images"
result_dir = "./dataset/results"

file_list = os.listdir(result_dir)
file_list = [f for f in file_list if ".jpg" in f]

sum=0
count=0
for file in file_list:
    original_path = os.path.join(data_dir, file)
    result_path = os.path.join(result_dir, file)

    ref_image_tensor = read_image(original_path)
    crop_tensor = read_image(result_path)

    crop_tensor=crop_tensor.to('cuda')
    ref_image_tensor=ref_image_tensor.to('cuda')
    result_feat = clip_model.encode_image(crop_tensor)
    ref_feat = clip_model.encode_image(ref_image_tensor)
    result_feat=result_feat.to('cpu').float()
    ref_feat=ref_feat.to('cpu').float()
    result_feat = result_feat / result_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    similarity = (100.0 * result_feat @ ref_feat.T)
    sum=sum+similarity.item()
    count=count+1
print(sum/count)
