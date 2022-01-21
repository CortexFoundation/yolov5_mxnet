#!/usr/bin/python
#coding:utf-8

import argparse
import os
import numpy as np
from mxnet import nd
from yolo import yolov5
import cv2
import mxnet as mx
from utils import non_max_suppression, Annotator, scale_coords, Colors

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=bool,  default=True,     help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,   default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,   default=1,         help="batch size used for training")
    parser.add_argument("--classes",     type=int,   default=80,        help="how many classes for the detection and classfication problem")
    parser.add_argument("--imgsz",       type=int,   default=640,       help="input image size")
    parser.add_argument("--dataset",     type=str,   default="./dataset/trial/images",   help="trial data for debug or training")
    parser.add_argument("--model_dir",   type=str,   default="./weights/",      help="Model dir for save and load")
    parser.add_argument("--model",       type=str,   default="yolov5m", help="model name")
    parser.add_argument("--silu",        type=str,   default="silu",    help="activation with silu or relu")
    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
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

    args = parse_opt()
    ctx = mx.cpu() if args.cpu else mx.gpu(args.gpu)

    gw = {"n":1, "s":2, "m":3, "l":4, "x":5}
    gd = {"n":1, "s":1, "m":2, "l":3, "x":4}
    postfix = args.model[-1]
    model = yolov5(batch_size=args.batch_size, mode="val", ctx=ctx, act=args.silu, gd=gd[postfix], gw=gw[postfix])
    model.collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
    #model.hybridize()

    try:
        EPOCH = []
        start_epoch = 0
        for f in os.listdir(args.model_dir):
            if f.endswith("params") and args.model in f:
                name_epoch = f.strip().split(".")[0].split("-")
                if len(name_epoch) == 2 and name_epoch[0] == args.model:
                    EPOCH.append(name_epoch[1])
        tmp = [int(_) for _ in EPOCH]
        ind = tmp.index(max(tmp))
        params_file = os.path.join(args.model_dir, args.model+"-"+EPOCH[ind]+".params")
        model.collect_params().load(params_file,ignore_extra=False)
        
        print(f'load weight {params_file} successfully')
    except:
        print("failed to load weight")

    dirs = os.path.join("./result/", opt.model) 
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        for f in os.listdir(dirs):
            os.remove(os.path.join(dirs,f))

    for f in os.listdir(args.dataset):
        _, ext = os.path.splitext(f)
        if ext != ".jpg" and ext != ".JPG" and ext != ".png" and ext != ".PNG":
            continue
        print(f)
        file_name = os.path.join(args.dataset, f)
        img = cv2.imread(file_name)
        img0s = img.copy()
        height, width = img.shape[0:2]
        scale = min(args.imgsz/height, args.imgsz/width)
        h0, w0 = height*scale, width*scale
        img0 = cv2.resize(img, (round(w0/32.)*32, round(h0/32.)*32))
        img = img0.astype("float32")/255.
        
        img = nd.array(img.transpose((2,0,1))[None], ctx = ctx)
        pred = model(img).asnumpy()
        
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        annotator = Annotator(img0s, line_width=1, example=str(names))
        
        det = pred[0]
        if det.shape[0] > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
        else:
            continue
        
        for *xyxy, conf, cls in reversed(det):
            c  =int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=Colors()(c, True))
        
        img0s = annotator.result()
        cv2.imwrite(os.path.join(dirs, f), img0s)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
