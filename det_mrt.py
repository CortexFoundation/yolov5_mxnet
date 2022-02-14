#!/usr/bin/python
#coding:utf-8

import argparse
import os
import numpy as np
from mxnet import nd
from yolo import yolov5
import cv2
import mxnet as mx
from mxnet import gluon
from utils import non_max_suppression, Annotator, scale_coords, Colors, str2bool, get_quantized_model, concat_out, make_squre
from mrt import sim_quant_helper as sim

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=str2bool,default=False,   help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,   default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,   default=1,         help="batch size used for training")
    parser.add_argument("--classes",     type=int,   default=80,        help="how many classes for the detection and classfication problem")
    parser.add_argument("--imgsz",       type=int,   default=640,       help="input image size")
    parser.add_argument("--dataset",     type=str,   default="./dataset/trial/images",   help="trial data for debug or training")
    parser.add_argument("--model_dir",   type=str,   default="./qout",      help="Model dir for save and load")
    parser.add_argument("--model",       type=str,   default="yolov5s", help="model name")
    parser.add_argument("--fuse",        type=str,   default=True,      help="activation with silu or relu")
    opt = parser.parse_args()
    return opt


def main(opt):
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    # class name 
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

    ctx = mx.cpu() if opt.cpu else mx.gpu(opt.gpu)
    print(ctx)
    
    qgraph, inputs_ext, oscales = get_quantized_model(opt.model_dir, opt.model, ctx)
   

    dirs = os.path.join("./results/", opt.model)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        for f in os.listdir(dirs):
            os.remove(os.path.join(dirs,f))

    for f in os.listdir(opt.dataset):
        _, ext = os.path.splitext(f)
        if ext != ".jpg" and ext != ".JPG" and ext != ".png" and ext != ".PNG":
            continue
        print(f)
        file_name = os.path.join(opt.dataset, f)
        _,_,_,_,img = make_squre(cv2.imread(file_name))
        img = cv2.resize(img, (opt.imgsz, opt.imgsz))
        img0s = img.copy()
        img = img.astype("float32")/255.

        img = nd.array(img.transpose((2,0,1))[None], ctx = ctx)
        img = sim.load_real_data(img, 'data', inputs_ext)
        out = qgraph(img)
        out = [t/oscales[i] for i,t in enumerate(out)]
        out = concat_out(*out).asnumpy()

        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
        annotator = Annotator(img0s, line_width=1, example=str(names))
        
        pred = out[0]
        if pred.shape[0] > 0:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0s.shape).round()
        
        for *xyxy, conf, cls in reversed(pred):
            c  =int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=Colors()(c, True))
        
        img0s = annotator.result()
        cv2.imwrite(os.path.join(dirs, f), img0s)

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
