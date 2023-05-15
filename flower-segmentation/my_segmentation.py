import os

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms

from model import ResNetUNet
from utilities import reverse_transform_mask


class FlowerSegmentor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ResNetUNet(1).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        image = self.trans(image)  # ToTensor
        image = transforms.Normalize(  # TODO: remove this?
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])(image)
        return image

    def segment(self, img):
        h, w = img.shape[:2]

        img = self.preprocess(img)
        img = img.unsqueeze(0).to(self.device)
        seg = self.model(img)[0]
        seg = torch.sigmoid(seg)
        seg_np = seg.cpu().detach()
        seg_np = reverse_transform_mask(seg_np)
        seg_np = np.where(seg_np > 220, 1, 0)
        seg_np = seg_np.astype("uint8") * 255
        seg_np = cv2.dilate(seg_np, np.ones((25, 25)))
        seg_np = cv2.resize(seg_np, (w, h))
        return seg_np


if __name__ == "__main__":
    segmentor = FlowerSegmentor("./model/pretrained/latest_weights.pth")
    img = cv2.imread("../dataset/test/data/0036f0f759621064.jpg")
    seg_np = segmentor.segment(img)
    cv2.imwrite("./test/sample.png", seg_np)
