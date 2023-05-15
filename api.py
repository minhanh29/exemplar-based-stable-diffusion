import os
import sys
import numpy as np
sys.path.append("./Paint-by-Example/")
sys.path.append("./flower-segmentation/")

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
from typing import List
from my_inpainting import ImageInpainting
from my_segmentation import FlowerSegmentor

app = FastAPI()

filenames = ["original.png", "ref.png"]
data_dir = "./temp"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
filepaths = [os.path.join(data_dir, file) for file in filenames]
mask_path = os.path.join(data_dir, "mask.png")
result_path = os.path.join(data_dir, "result.png")

model = ImageInpainting("./checkpoints/my_sd_model.ckpt")
segmentor = FlowerSegmentor("./checkpoints/segmentation.pth")


@app.post("/edit_image")
async def edit_image(files: List[UploadFile] = File(...)):
    if len(files) != 2:
        return {
            "Error": "There must be 2 files."
        }

    for file, destination_file_path in zip(files, filepaths):
        async with aiofiles.open(destination_file_path, 'wb') as out_file:
            while content := await file.read(1024):  # async read file chunk
                await out_file.write(content)  # async write file chunk

    # extract flower mask
    print("Extracting mask...")
    img = cv2.imread(filepaths[0])
    mask = segmentor.segment(img)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask]*3, axis=-1)
    cv2.imwrite(mask_path, mask)

    # run the image inpainting job
    print("Run inpainting...")
    model.predict(filepaths[0], mask_path, filepaths[1], result_path)
    print("Done.")

    return FileResponse(result_path)

