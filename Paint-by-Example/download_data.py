import fiftyone

dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              classes=["Flower"],
    dataset_dir="./dataset/open-images/",
              # max_samples=50000,
          )
