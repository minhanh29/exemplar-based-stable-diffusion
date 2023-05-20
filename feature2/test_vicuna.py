"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args


# @torch.inference_mode()
# def main(args):
#     model, tokenizer = load_model(
#         args.model_path,
#         args.device,
#         args.num_gpus,
#         args.max_gpu_memory,
#         args.load_8bit,
#         args.cpu_offloading,
#         debug=args.debug,
#     )

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     msg = args.message

#     conv = get_conversation_template(args.model_path)
#     conv.append_message(conv.roles[0], msg)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     input_ids = tokenizer([prompt]).input_ids
#     output_ids = model.generate(
#         torch.as_tensor(input_ids).cuda(),
#         do_sample=True,
#         temperature=args.temperature,
#         max_new_tokens=args.max_new_tokens,
#     )
#     if model.config.is_encoder_decoder:
#         output_ids = output_ids[0]
#     else:
#         output_ids = output_ids[0][len(input_ids[0]) :]
#     outputs = tokenizer.decode(
#         output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
#     )

#     print(f"{conv.roles[0]}: {msg}")
#     print(f"{conv.roles[1]}: {outputs}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     add_model_args(parser)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--max-new-tokens", type=int, default=512)
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--message", type=str, default="Hello! Who are you?")
#     args = parser.parse_args()

#     main(args)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./checkpoints/my_vicuna_7b/"
model, tokenizer = load_model(
    model_path,
    device, num_gpus=1
)

# model = model.to(device)
flower_name = "rose"
flower_color = "white"
msg = f"Give 4 detailed prompts to generate realistic images of only {flower_color} {flower_name} of different locations (in a vase, pot, or bouquet, and so on), appearance, and arrangement. The prompt only contains keywords separated by comma. One prompt each line without any markers."

conv = get_conversation_template(model_path)
conv.append_message(conv.roles[0], msg)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=0.1,
    max_new_tokens=512,
)
if model.config.is_encoder_decoder:
    output_ids = output_ids[0]
else:
    output_ids = output_ids[0][len(input_ids[0]) :]
outputs = tokenizer.decode(
    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
)

print(f"{conv.roles[0]}: {msg}")
print(f"{conv.roles[1]}: {outputs}")
