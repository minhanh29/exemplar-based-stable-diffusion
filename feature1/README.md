# Machine Learning Project - Group 07

**Team Members**
1. Nguyen Minh Anh  
2. Nguyen Thanh Sang  
3. Nguyen Bao Han  
4. Tran Thuc Ai Quynh  

## System Requirements
- An Nvidia GPU with at least 40GB VRAM
- CUDA 11.3+
- 50GB disk space
- Azcopy installed (Download from "")

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

3. Download the model weights (Install Azcopy to download the models, see **System Requitements** section)
```
python download_model.py
```


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

3. Download the model weights
```
python download_model.py
```

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


## Model Training
### Image Editing Model
1. Go to the Paint-by-Example folder
```
cd Paint-by-Example
```
2. Download the data for training
```
python download_data.py
```
3. Filter data and create bounding boxes
```
python scripts/read_bbox.py
```
4.  Download the stable diffusion 1.5 weights on the official website and put it to the checkpoints folder
5. Start training
```
bash train.sh
```
6. The trained checkpoints are located under "models/Paint-by-Example/{timestamp}_v1/checkpoints"


### Image Segmentation Model
1. Go to the flower-segmentation folder
```
cd flower-segmentation
```
2. Download the data for training
```
python download_data.py
```
3. Create mask for the 102Flower dataset
