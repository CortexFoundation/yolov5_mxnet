## Introduction
This repo includes training yolov5 model in MXNET framework, model quantization and conversion on Cortex CVM Runtime mothodology. The code is revised from https://github.com/ultralytics/yolov5 as MXNET version, so as to do Cortex MRT quantization which could implement deterministic on-chain inference on fixed-point model uploaded.
## How to use
### Train your own model
Download this repo to local machine

git clone https://github.com/CortexFoundation/yolov5_mxnet.git

Prepare the training dataset including image files and corresponding labels, make the directory structure as following,

dataset
  |
  |
  |----train
         |
         |
         |----images
                |
                |------img000.jpg
                |------img001.jpg
                |    ...
                |------imgxxx.jpg
         
         
  
  
