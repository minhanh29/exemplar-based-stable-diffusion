import os
import sys
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from time import time
from PIL import Image
import torch
from fastchat.model import load_model, get_conversation_template, add_model_args

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
from typing import List

app = FastAPI()

data_dir = "./temp"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)


device = "cuda" if torch.cuda.is_available() else "cpu"
original_path = os.path.join(data_dir, "original.png")
processor = AutoProcessor.from_pretrained("microsoft/git-large-textvqa")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textvqa")
git_model = git_model.to(device)

vicuna_model_path = "./checkpoints/my_vicuna_7b/"
vicuna_model, tokenizer = load_model(
    vicuna_model_path,
    device, num_gpus=1
)

model_id = "runwayml/stable-diffusion-v1-5"
sd_model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
sd_model = sd_model.to(device)


def generate_prompts(flower_info):
    msg = f"Give 4 detailed prompts to generate realistic images of only {flower_info} of different locations (in a vase, pot, or bouquet, and so on), appearance, and arrangement. The prompt only contains keywords separated by comma. Show each prompt on a separate line with maximum 20 words each."

    conv = get_conversation_template(vicuna_model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = vicuna_model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.01,
        max_new_tokens=512,
    )
    if vicuna_model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    outputs = outputs.replace("Prompt: ", "")
    for i in range(1, 5):
        outputs = outputs.replace(f"{i}. ", "")
    outputs = outputs.split("\n")
    result = [line for line in outputs if line.strip() != ""]
    return result



def extract_flower_info(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    question = "what is the name of the flower?"

    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    generated_ids = git_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    flower_name = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    flower_name = flower_name.replace(question, "")

    question = "the colors of the flower are"

    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    generated_ids = git_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    flower_color = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    flower_color = flower_color.replace(question, "")

    return f"{flower_color}{flower_name}".strip()


@app.post("/get_image")
def get_image(id: int):
    return FileResponse(os.path.join(data_dir, f"{id}.png"))


@app.get("/generate_image")
def generate_image():
    # extract flower mask
    flower_info = extract_flower_info(original_path)

    prompts = generate_prompts(flower_info)
    print(prompts)
    images = sd_model(prompts).images

    for i, image in enumerate(images):
        image.save(os.path.join(data_dir, f"{i}.png"))
    return {
        "success": True
    }
