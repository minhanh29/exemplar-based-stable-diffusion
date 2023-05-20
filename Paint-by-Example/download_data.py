import fiftyone
import os

dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["detections"],
              classes=["Flower"],
            dataset_dir="./dataset/open-images/",
          )
dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="test",
              label_types=["detections"],
              classes=["Flower"],
            dataset_dir="./dataset/open-images/",
          )
dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              classes=["Flower"],
            dataset_dir="./dataset/open-images/",
          )


# download the annotation files
os.system("azcopy copy '' './dataset/open-images/annotations/' --recursive")
