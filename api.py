import os
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
from typing import List
from my_inference import Inference

app = FastAPI()

filenames = ["original.png", "mask.png", "ref.png"]
data_dir = "./temp"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
filepaths = [os.path.join(data_dir, file) for file in filenames]
result_path = os.path.join(data_dir, "result.png")

model = Inference("./checkpoints/my_sd_model.ckpt")
# model = Inference("./checkpoints/model.ckpt")

@app.post("/edit_image")
async def edit_image(files: List[UploadFile] = File(...)):
    if len(files) != 3:
        return {
            "Error": "There must be 3 files."
        }

    for file, destination_file_path in zip(files, filepaths):
        async with aiofiles.open(destination_file_path, 'wb') as out_file:
            while content := await file.read(1024):  # async read file chunk
                await out_file.write(content)  # async write file chunk

    # run the image inpainting job
    model.predict(filepaths[0], filepaths[1], filepaths[2], result_path)

    return FileResponse(result_path)

