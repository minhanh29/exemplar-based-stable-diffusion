python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model ./models/Paint-by-Example/2023-05-02T17-23-29_v1/checkpoints/last.ckpt \
--resume ./models/Paint-by-Example/2023-05-02T17-23-29_v1/ \
--base configs/v1.yaml \
--scale_lr False
