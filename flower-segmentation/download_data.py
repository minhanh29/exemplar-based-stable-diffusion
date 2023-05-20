import fiftyone
import os

# Open Image dataset
dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["segmentations"],
              classes=["Flower"],
            dataset_dir="./dataset/",
              # max_samples=10,
          )


# download the 102Flower Dataset

