import os


# eval data for image editing models
os.system(f"azcopy copy '' './Paint-by-Example/dataset/'")

# eval data for segmentation models
os.system(f"azcopy copy '' './flower-segmentation/dataset/'")
