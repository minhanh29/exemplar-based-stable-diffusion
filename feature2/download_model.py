import os

# download converted vicuna 7B model
os.system(f"azcopy copy '' './checkpoints/my_vicuna_7b/' --recursive")
