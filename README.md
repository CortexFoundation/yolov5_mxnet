## Introduction
This repo includes training yolov5 model in MXNET framework, model quantization and conversion on Cortex CVM Runtime mothodology. The code is revised from https://github.com/ultralytics/yolov5 as MXNET version, so as to do Cortex MRT quantization which could implement deterministic on-chain inference on fixed-point model uploaded.
## How to use
### Train your own model
Download this repo to local machine

    git clone https://github.com/CortexFoundation/yolov5_mxnet.git 

Prepare the training dataset including image files and corresponding labels, make the directory structure as following,

![dataset directory structure](https://github.com/CortexFoundation/yolov5_mxnet/blob/main/src/tree.jpg)

For training, one need to specify the model name, batch size learning rate etc, then type command below for an example.

    python train.py --model=yolov5x --cpu=False --gpu=0 --batch_size=16 --lr=0.0001

Two traned model files (.params and .json) will be generated in folder ./weight/, which can used for validatation via 

    python detect.py --model=yolov5x --cpu=False --gpu=0 --batch_size=1

and the final output images save in ./results/yolov5x/ directory.

The model accuracy in training depends on datasets (number of images, object balance, class balance, ...) and learning rate (optimizer, initial value, ...), so one need great care while training to get a satisfactory training model.

### CVM Runtime and MRT
#### cvm-runtime

The CVM Runtime library is used in @CortexLabs full-node project: CortexTheseus, working for pure deterministic AI model inference.

First download Cortex CVM-Runtime repo https://github.com/CortexFoundation/cvm-runtime.git

    git clone https://github.com/CortexFoundation/cvm-runtime.git

and compile

    cd cvm-runtime
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

Please refer to https://github.com/CortexFoundation/cvm-runtime/blob/master/docs/cvm/install.md for details.

#### MRT

MRT, short for Model Representation Tool, aims to convert floating model into a deterministic and non-data-overflow network. MRT links the off-chain developer community to the on-chain ecosystem, from Off-chain deep learning to MRT transformations, and then uploading to Cortex Blockchain for on-chain deterministic inference.

There is a detailed example https://github.com/CortexFoundation/cvm-runtime/blob/dev-example/docs/mrt/example.md for MRT usage. After one got a trained model,

    cp path-to-model-params/yolov5x-0010.params path-to-model-for-mrt/yolov5x.params
    cp path-to-model-json/yolov5x-0010.json path-to-model-for-mrt/yolov5x.json

    


