import os
from torchmetrics import JaccardIndex
import glob
from imutils import paths
import imageio

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ResNetUNet, dice_loss
from utilities import reverse_transform, reverse_transform_mask
from preprocess import check_dir

DISPLAY_PLOTS = False
TEST_DIR = "../../segmentation/dataset_small/jpg"
SAVE_PATH = "./test/output"
PREFIX = "seg_"

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])


class parseTestset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        _, filename = os.path.split(image_path)

        if self.transform:
            image = self.transform(image)  # ToTensor
            image = transforms.Normalize(  # TODO: remove this?
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])(image)
        return image, filename

class parseDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths, self.mask_paths = img_paths, mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.img_paths[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))

        mask = self.mask_paths[idx]
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (300, 300))
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # for images but not for masks
            image = transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])(image)

        return [image, mask]

def cal_iou(pred, target, n_classes = 1):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on: {device}')

    num_class = 1
    model = ResNetUNet(num_class).to(device)

    model.load_state_dict(torch.load("../checkpoints/segmentation.pth"))
    # model.load_state_dict(torch.load("./model/pretrained/latest_weights.pth"))

    # test_img_paths = list(paths.list_images(TEST_DIR))
    mask_paths = glob.glob("../../segmentation/dataset_small/mask/*.png")
    mask_paths = mask_paths[:800]
    test_img_paths = list(map(lambda st: st.replace(".png", ".jpg").replace("mask", "jpg"), mask_paths))
    print(f'found {len(test_img_paths)} images')

    # small batch_size if you are testing on 1 or 2 images
    b_size = 25

    test_set = parseDataset(test_img_paths, mask_paths, transform=trans)
    test_loader = DataLoader(test_set, batch_size=b_size,
                             shuffle=True, num_workers=0)

    check_dir(SAVE_PATH)

    model.eval()
    iou_score = 0
    dice_score = 0
    cnt = 0
    jaccard = JaccardIndex(task="binary", num_class=1)
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        outputs = model(inputs)
        dice = dice_loss(torch.nn.functional.sigmoid(outputs), labels).item()

        outputs = outputs.detach().cpu().numpy().squeeze(1)
        outputs = (outputs > 0.5).astype("int32")
        outputs = torch.from_numpy(outputs)
        labels = labels.detach().cpu().numpy().squeeze(1)
        labels = (labels > 0.5).astype("int32")
        labels = torch.from_numpy(labels)
        iou = jaccard(outputs, labels).item()
        iou_score += iou
        dice_score += dice
        cnt += 1
        print(iou)

    print("IOU", iou_score/cnt)
    print("Dice", dice_score/cnt)
