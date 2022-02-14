## Introduction
This repo includes training yolov5 model in MXNET framework, model quantization and conversion on Cortex CVM Runtime mothodology. The code is revised from https://github.com/ultralytics/yolov5 as MXNET version, so as to do Cortex MRT quantization which could implement deterministic on-chain inference on fixed-point model uploaded.
## How to use

### Convert from Pytorch Model
Pytorch model weights are extraced and save as yolov5*.npy. This repo provided script convert.py in ./weights/ directory that convert pre-train Pytorch model to MXNet models *.params and *.json saveed in ./weights.

	python convert.py --model=yolov5n --fuze=True --cpu=True
	python convert.py --model=yolov5s --fuze=True --cpu=True
	python convert.py --model=yolov5m --fuze=True --cpu=True
	python convert.py --model=yolov5l --fuze=True --cpu=True
	python convert.py --model=yolov5x --fuze=True --cpu=True

### Train your own model
Download this repo to local machine

    git clone https://github.com/CortexFoundation/yolov5_mxnet.git 

Prepare the training dataset including image files and corresponding labels, make the directory structure as following,

![dataset directory structure](https://github.com/CortexFoundation/yolov5_mxnet/blob/main/src/tree.jpg)

For training, one need to specify the model name, batch size learning rate etc, then type command below for an example.

    python train.py --model=yolov5* --fuse=False --cpu=False --gpu=0 --batch_size=16 --lr=0.0001

Two traned model files (.params and .json) will be generated in folder ./weight/, which can used for validatation via 

    python detect.py --model=yolov5* --fuse=False --cpu=False --gpu=0 --batch_size=1

and the final output images save in ./results/yolov5*/ directory.

The model accuracy in training depends on datasets (number of images, object balance, class balance, ...) and learning rate (optimizer, initial value, ...), so one need great care while training to get a satisfactory training model.

### CVM Runtime and MRT
#### cvm-runtime

The CVM Runtime library is used in @CortexLabs full-node project: CortexTheseus, working for pure deterministic AI model inference.

First download Cortex CVM-Runtime repo https://github.com/CortexFoundation/cvm-runtime.git

    git clone https://github.com/CortexFoundation/cvm-runtime.git -t ryt

and compile

    cd cvm-runtime
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

Please refer to https://github.com/CortexFoundation/cvm-runtime/blob/master/docs/cvm/install.md for details.

After compiling and installation stage, one needs to configure the environment variable as following for example.
    export CVM_HOME=$HOME/cvm-runtime
    export PYTHONPATH=$CVM_HOME/python:${PYTHONPATH}
    export MRT_HOME=$CVM_HOME/python/mrt
    export PYTHONPATH=$MRT_HOME:${PYTHONPATH}

#### MRT

MRT, short for Model Representation Tool, aims to convert floating model into a deterministic and non-data-overflow network. MRT links the off-chain developer community to the on-chain ecosystem, from Off-chain deep learning to MRT transformations, and then uploading to Cortex Blockchain for on-chain deterministic inference.

There is a detailed example https://github.com/CortexFoundation/cvm-runtime/blob/dev-example/docs/mrt/example.md for MRT usage. After one got a trained model,

    cp weight/yolov5x-xxxx.params qout/yolov5x.params
    cp weight/yolov5x.json qout/yolov5*.json

make a yaml file yolov5*.yaml like as https://github.com/CortexFoundation/cvm-runtime/blob/ryt/tests/mrt/yolov5s/yolov5s.yaml, save as qconf/yolov5*.yaml (there are examples for different network structure), then run

    python quantize.py --model=yolov5*

MRT will generate calibration and quantization files, and also some json configure information files. The 

MRT will generate two files named "params" and "symbol" in qout/yolov5*_cvm/, which includes the quantized model parameters and symbol informations for model uploading.

#### Performance

Furthermore, we can examine performance degradation on quantization compared with float-point model,

    python det_mrt.py --model=yolov5*                # with quantization
    python detetect.py --model=yolov5* --fuse=False  # withou quantization

generates detection images in result/yolov5*/ with inferred from quantized or non-quantized model.

    python val_mrt.py --model=yolov5*                # with quantization
    python val.py --model=yolov5* --fuse=False       # withou quantization

will validate MAP performance and generate two files result/yolov5*_eval_quant.txt and result/yolov5*_eval_float.txt for comparison

we quantized the original models pre-trained in project https://github.com/ultralytics/yolov5, and found the performance loss is negligible for all model scales. Evaluation results are recorded in files "result/model-name_eval_quant.txt" and "result/model-name_eval_float.txt".

#### Model upload

Following instructions on https://github.com/CortexFoundation/torrentfs, to make torrent file and upload to Cortex blockchain. One can find the model in cerebro explorer https://cerebro.cortexlabs.ai/#/model and enjoy the on-chain AI inference after successful uploading.


    

