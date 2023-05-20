import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
import argparse
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


class FlowerDetection:
    def __init__(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    def detect(self, img_path, save_path=None):
        TEXT_PROMPT = "flower"
        BOX_TRESHOLD = 0.3
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(img_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        if save_path is not None:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            Image.fromarray(annotated_frame).save(save_path)

        result = []
        for box in boxes:
            x, y, w, h = box
            x1 = x - w/2
            x2 = x1 + w
            y1 = y - h/2
            y2 = y1 + h

            # expand the box
            x1 = min(max(x1 - 0.05, 0), 1)
            x2 = min(max(x2 + 0.05, 0), 1)
            y1 = min(max(y1 - 0.05, 0), 1)
            y2 = min(max(y2 + 0.05, 0), 1)

            x1 = int(x1 * image_source.shape[1])
            x2 = int(x2 * image_source.shape[1])
            y1 = int(y1 * image_source.shape[0])
            y2 = int(y2 * image_source.shape[0])
            result.append([x1, y1, x2, y2])
        print(result)
        return result

if __name__ == "__main__":
    detector = FlowerDetection()
    detector.detect("./temp/original.png", save_path="./my_dino.png")
