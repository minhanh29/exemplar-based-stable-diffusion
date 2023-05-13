python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model ./checkpoints/sd-v1-5-inpainting.ckpt \
--resume ./models/Paint-by-Example/2023-05-03T18-53-26_v1/checkpoints/last.ckpt \
--base configs/v1.yaml \
--no-test \
--scale_lr False
