# Machine Learning Project - Group 07

**Team Members**
1. Nguyen Minh Anh - s3927589
2. Nguyen Thanh Sang - s3894290 
3. Nguyen Bao Han - s3872104 
4. Tran Thuc Ai Quynh - s3878340 

## System Requirements
- An Nvidia GPU with at least 40GB VRAM
- CUDA 11.3+
- 150GB free disk space

## Installation
### Feature 1 - Flower Replacement
1. Go to the feature1/Paint-by-Example folder
```
cd feature1/Paint-by-Example
```

2. Create a virtual environment and install dependencies
```
conda env create -f environment.yaml
conda activate task2
```

3. Download the weights for the image editing model [here](https://mlrmit.blob.core.windows.net/models/my_sd_model.ckpt) and the segmentation model [here](https://mlrmit.blob.core.windows.net/models/segmentation_adamw.pth) and put all of them under the **feature1/checkpoints** folder. 
The folder structure should look like this

```
project
├── feature1
│  ├── checkpoints
│  │  ├── my_sd_model.ckpt
│  │  ├── segmentation_adamw.pth
```

4. Install the GroundDINO library (For flower detection)
```
cd feature1/GroundingDINO
python -m pip install -e GroundingDINO
```

5. The GroundingDINO weights will be automatically downloaded when you first run the API

### Feature 2 - Flower Style Variations
1. From root folder, go to the feature2 folder
```
cd feature2
```

2. Create a different virtual environment and install dependencies
```
conda env create -n task2 python=3.9
conda activate task2
pip install -r requirements.txt
```

3. Download the final Vicuna model weights [here](https://mlrmit.blob.core.windows.net/models/my_vicuna_7b.zip) (zip file) and extract it to the feature2/checkpoints folder. (The ready-to-use Vicuna model is not available not the Internet. It must be produced by applying Vicuna delta weight to the original LLama 7B weight from Huggingface).  
The folder structure should look like this
```
project
├── feature2
│  ├── checkpoints
│  │  ├── my_vicuna_7b
```

4. The GIT and Stable Diffusion model will be automatically downloaded from HuggingFace when you first run the application

### Frontend 
1. Install NodeJS from their released website https://nodejs.org/en/blog/release/v14.20.0
2. Install yarn
```
npm install -g yarn
```
3. Go to the frontend folder
```
cd frontend
```
4. Install dependencies
```
yarn install
```

## Run the Demo
### Start the backend
1. Open a terminal session, go to feature2 folder and start the FastAPI application as follow
```
cd feature2
conda activate task2
uvicorn api:app --port 8000
```

2. Open **another** terminal session, go to feature1 folder and start the FastAPI application as follow
```
cd feature1
conda activate task1
uvicorn api:app --host 0.0.0.0 --port 8080
```

### Start the frontend
1. Go to the frontend folder and run the following command
```
yarn start
```
2. Go to **localhost:3000** on your brower and start testing the system.


## Model Training and Evaluating
### Image Editing Model
1. Go to the Paint-by-Example folder and activate task1 environment
```
cd feature1/Paint-by-Example
conda activate task1
```
2. Download the image_editing_dataset.zip from [here](https://mlrmit.blob.core.windows.net/models/image_editing_dataset.zip) and extract it in the current folder.
The folder structure should look like this
```
project
├── feature1
│  ├── Paint-by-Example
│  │  ├── dataset
│  │  │  ├── open-images
│  │  │  ├── test_benchmark
```

4.  Download the pretrained Paint-by-Example checkpoint from [here](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt) and put it to the checkpoints folder
The folder structure should look like this
```
project
├── feature1
│  ├── Paint-by-Example
│  │  ├── checkpoints
│  │  │  ├── model.ckpt
```

5. Run the following command to start training
```
python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model ./checkpoints/model.ckpt \
--base configs/v1.yaml \
--no-test \
--scale_lr False
```
6. The trained checkpoints are located under the **models/Paint-by-Example/{timestamp}_v1/checkpoints** folder
  
7. To evaluate the model on the test benchmark dataset, first, run the following command to generate images based on the input from the test dataset.
```
python scripts/inference_test_bench.py \
--plms \
--outdir results/test_bench \
--config configs/v1.yaml \
--ckpt [path to the trained .ckpt file] \
--scale 5
```
8. Compute the FID score
```
python eval_tool/fid/fid_score.py --device cuda \
test_bench/test_set_GT \
results/test_bench/results
```
9. Compute the QS Score. First, download the model weights for the QS Score from [Google Drive](https://drive.google.com/file/d/1Ce2cSQ8UttxcEk03cjfJgaBwdhSPyuHI/view) and save it under the eval_tool/gmm. Then run
```
python eval_tool/gmm/gmm_score_coco.py results/test_bench/results \
--gmm_path eval_tool/gmm/coco2017_gmm_k20 \
--gpu 1
```
10. Compute the CLIP score
```
python eval_tool/clip_score/region_clip_score.py \
--result_dir results/test_bench/results
```


### Image Segmentation Model
1. Go to the flower-segmentation folder and activate task1 environment
```
cd feature1/flower-segmentation
conda activate task1
```
2. Download the segmentation_dataset.zip from [here](https://mlrmit.blob.core.windows.net/models/segmentation_dataset.zip) and extract it under the current folder
The folder structure should look like this
```
project
├── feature1
│  ├── flower-segmentation
│  │  ├── dataset
│  │  │  ├── test_bench
│  │  │  ├── train
```
3. Run the following command to start training
```
python train.py
```
4. The trained model will be saved under the **model/pretrained** folder
5. To evaluate the model, run the evaluate.py script and provide the model path
```
python evaluate.py [path to the .pth file]
```

### Flower Style Variations
1. Go to the feature2 folder and activate the environment
```
cd feature2
conda activate task2
```
2. Download the test dataset [here](https://mlrmit.blob.core.windows.net/models/feature2_dataset.zip). Extract it to the current folder
The folder structure should look like this
```
project
├── feature2
│  ├── dataset
│  │  ├── images
│  │  │  ├── image_00001.jpg
│  │  │  ├── image_00002.jpg
...
```
3. Run the following script to generate images using the framework
```
python generate_images.py
```
4. To evaluate the model, run the clip_score.py script 
```
python clip_score.py
```