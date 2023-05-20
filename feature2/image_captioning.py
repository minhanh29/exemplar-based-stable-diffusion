from transformers import AutoProcessor, AutoModelForCausalLM
from time import time
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("microsoft/git-large-textvqa")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

st = time()
image = Image.open("../feature1/temp/ref.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

question = "the name of the flower is"

input_ids = processor(text=question, add_special_tokens=False).input_ids
input_ids = [processor.tokenizer.cls_token_id] + input_ids
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
flower_name = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
flower_name = flower_name.replace(question, "")

question = "the colors of the flower are"

input_ids = processor(text=question, add_special_tokens=False).input_ids
input_ids = [processor.tokenizer.cls_token_id] + input_ids
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
flower_color = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
flower_color = flower_color.replace(question, "")

print(flower_name, flower_color)
print(time() - st)
