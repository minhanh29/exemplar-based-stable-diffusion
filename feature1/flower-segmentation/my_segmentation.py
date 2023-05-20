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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def segment_single(self, img):
        h, w = img.shape[:2]

        img = self.preprocess(img)
        img = img.unsqueeze(0).to(self.device)
        seg = self.model(img)[0]
        seg = torch.sigmoid(seg)
        seg_np = seg.cpu().detach()
        seg_np = reverse_transform_mask(seg_np)
        seg_np = np.where(seg_np > 220, 1, 0)
        seg_np = seg_np.astype("uint8") * 255
        # seg_np = cv2.dilate(seg_np, np.ones((25, 25)))
        seg_np = cv2.resize(seg_np, (224, 224))
        return seg_np

    def segment_batch(self, images):
        images = images.to(self.device)
        seg = self.model(images)
        seg = torch.sigmoid(seg)
        seg_np = seg.cpu().detach()

        seg_np = seg_np.numpy().transpose((0, 2, 3, 1))
        seg_np = np.clip(seg_np, 0, 1)
        seg_np = (seg_np * 255).astype(np.uint8)
        seg_np = np.where(seg_np > 220, 1, 0)
        seg_np = seg_np.astype("uint8") * 255
        return seg_np

    def segment(self, img, boxes=[]):
        h, w = img.shape[:2]
        if len(boxes) == 0:
            seg = self._segment(img, dilate=True)
            mask = cv2.dilate(mask, np.ones((5, 5)))
            seg = cv2.resize(seg, (w, h))
            return seg

        mask = np.zeros((h, w), dtype="uint8")
        for x1, y1, x2, y2 in boxes:
            crop_img = img[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = np.maximum(self._segment(crop_img, dilate=True), mask[y1:y2, x1:x2])
        mask = cv2.dilate(mask, np.ones((5, 5)))
        # mask = cv2.resize(mask, (w, h))
        return mask

    def _segment(self, img, dilate=True):
        h, w = img.shape[:2]
        mask = self.segment_single(img) #scale = 1
        score = np.mean(mask)

        # multiscale segmentation
        print("Mask score", score)
        if score < 10:
            print("Do multiscale")
            mask = np.maximum(mask, self._segment_tile(img, scale=2))
            mask = np.maximum(mask, self._segment_tile(img, scale=3))
            mask = np.maximum(mask, self._segment_tile(img, scale=5))
            if dilate:
                mask = cv2.dilate(mask, np.ones((19, 19)))
        elif dilate:
            mask = cv2.dilate(mask, np.ones((23, 23)))
        mask = cv2.resize(mask, (w, h))
        return mask

    def _segment_tile(self, img, scale=3, step_scale=4):
        h, w = img.shape[:2]
        window_size = 224
        size = window_size * scale
        if h > w:
            img = cv2.resize(img, (size, int(h * size / w)))
        else:
            img = cv2.resize(img, (int(w * size / h), size))

        step_size = window_size // step_scale
        batch = []
        for i in range(step_scale * (scale - 1) + 1):
            start_h = i  * step_size
            end_h = start_h + window_size
            for j in range(step_scale * (scale - 1) + 1):
                start_v = j  * step_size
                end_v = start_v + window_size
                processed = self.preprocess(img[start_h:end_h, start_v:end_v].copy())
                batch.append(processed.unsqueeze(0))

        batch = torch.cat(batch, dim=0)
        seg = self.segment_batch(batch)
        mask = np.zeros((size, size, 1))
        cnt = 0
        for i in range(step_scale * (scale - 1) + 1):
            start_h = i  * step_size
            end_h = start_h + window_size
            for j in range(step_scale * (scale - 1) + 1):
                start_v = j  * step_size
                end_v = start_v + window_size
                mask[start_h:end_h, start_v:end_v] = np.maximum(mask[start_h:end_h, start_v:end_v], seg[cnt])
                cnt += 1
        mask = mask.astype("uint8")
        mask = cv2.resize(mask, (224, 224))
        return mask


if __name__ == "__main__":
    segmentor = FlowerSegmentor("../checkpoints/segmentation_adamw.pth")
    # segmentor = FlowerSegmentor("./model/pretrained/latest_weights.pth")
    # img = cv2.imread("../dataset/test/data/0036f0f759621064.jpg")
    # img = cv2.imread("./test/yellow_rose.jpg")
    # img = cv2.imread("./test/roses.jpg")
    img = cv2.imread("../temp/original.png")
    # img = cv2.imread("./test/two.jpg")
    # img = cv2.imread("./test/pink.jpg")
    # seg_np = segmentor.segment(img, boxes=[[0, 3, 546, 276], [106, 8, 186, 66], [341, 3, 423, 63], [373, 22, 466, 87], [179, 24, 265, 89], [428, 184, 524, 271], [467, 103, 546, 172], [460, 79, 543, 133]])
    seg_np = segmentor.segment(img)
    cv2.imwrite("./test/mask.png", seg_np)
