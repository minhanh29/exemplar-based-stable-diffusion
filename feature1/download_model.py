import os

checkpoint_dir = "./checkpoints"

# download segmentation model
os.system(f"azcopy copy '' {checkpoint_dir}/segmentation.pth")

# download image editing model
os.system(f"azcopy copy '' {checkpoint_dir}/my_sd_model.ckpt")
