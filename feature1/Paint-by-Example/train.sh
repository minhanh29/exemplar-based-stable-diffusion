python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model ./checkpoints/sd-v1-5-inpainting.ckpt \
--base configs/v1.yaml \
--no-test \
--scale_lr False
