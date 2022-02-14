#!/usr/bin/python
#coding:utf-8

import argparse
import os
import numpy as np
from mxnet import nd
from yolo import yolov5
import cv2
from utils import str2bool
import mxnet as mx
from utils import non_max_suppression, scale_coords, xywh2xyxy, process_batch, ap_per_class, make_squre

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=str2bool, default=False,    help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,     default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,     default=1,         help="batch size used for training")
    parser.add_argument("--classes",     type=int,     default=80,        help="how many classes for the detection and classfication problem")
    parser.add_argument("--imgsz",       type=int,     default=640,       help="input image size")
    parser.add_argument("--dataset",     type=str,     default="./dataset/trial/images",   help="trial data for debug or training")
    parser.add_argument("--model_dir",   type=str,     default="./weights/",      help="Model dir for save and load")
    parser.add_argument("--model",       type=str,     default="yolov5s", help="model name")
    parser.add_argument("--fuse",        type=str2bool,default=False,    help="fuse conv and normal")
    opt = parser.parse_args()
    return opt


def main(opt):
    conf_thres = 0.001
    iou_thres = 0.6
    # class name 
    names = {0 : 'person', 1 : 'bicycle', 2 : 'car', 3 : 'motorcycle', 4 : 'airplane', 5 : 'bus', 6 : 'train', 7 : 'truck', 
             8 : 'boat', 9 : 'traffic light', 10 : 'fire hydrant', 11 : 'stop sign', 12 : 'parking meter', 13 : 'bench', 14 : 'bird', 15 : 'cat', 
             16 : 'dog', 17 : 'horse', 18 : 'sheep', 19 : 'cow', 20 : 'elephant', 21 : 'bear', 22 : 'zebra', 23 : 'giraffe', 
             24 : 'backpack', 25 : 'umbrella', 26 : 'handbag', 27 : 'tie', 28 : 'suitcase', 29 : 'frisbee', 30 : 'skis', 31 : 'snowboard', 
             32 : 'sports ball', 33 : 'kite', 34 : 'baseball bat', 35 : 'baseball glove', 36 : 'skateboard', 37 : 'surfboard', 38 : 'tennis racket', 39 : 'bottle', 
             40 : 'wine glass', 41 : 'cup', 42 : 'fork', 43 : 'knife', 44 : 'spoon', 45 : 'bowl', 46 : 'banana', 47 : 'apple', 
             48 : 'sandwich', 49 : 'orange', 50 : 'broccoli', 51 : 'carrot', 52 : 'hot dog', 53 : 'pizza', 54 : 'donut', 55 : 'cake', 
             56 : 'chair', 57 : 'couch', 58 : 'potted plant', 59 : 'bed', 60 : 'dining table', 61 : 'toilet', 62 : 'tv', 63 : 'laptop', 
             64 : 'mouse', 65 : 'remote', 66 : 'keyboard', 67 : 'cell phone', 68 : 'microwave', 69 : 'oven', 70 : 'toaster', 71 : 'sink', 
             72 : 'refrigerator', 73 : 'book', 74 : 'clock', 75 : 'vase', 76 : 'scissors', 77 : 'teddy bear', 78 : 'hair drier', 79 : 'toothbrush'}

    ctx = mx.cpu() if opt.cpu else mx.gpu(opt.gpu)
    print(ctx)

    gw = {"n":1, "s":2, "m":3, "l":4, "x":5}
    gd = {"n":1, "s":1, "m":2, "l":3, "x":4}
    postfix = opt.model[-1]
    model = yolov5(batch_size=opt.batch_size, mode="val", ctx=ctx, gd=gd[postfix], gw=gw[postfix], fuse=opt.fuse)
    model.collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
    #model.hybridize()

    try:
        EPOCH = []
        for f in os.listdir(opt.model_dir):
            if f.endswith("params") and opt.model in f:
                name_epoch = f.strip().split(".")[0].split("-")
                if len(name_epoch) == 2 and name_epoch[0] == opt.model:
                    EPOCH.append(name_epoch[1])
        tmp = [int(_) for _ in EPOCH]
        ind = tmp.index(max(tmp))
        params_file = os.path.join(opt.model_dir, opt.model+"-"+EPOCH[ind]+".params")
        model.collect_params().load(params_file,ignore_extra=False)
        
        print(f'load weight {params_file} successfully')
    except:
        print("failed to load weight")

    iouv = np.linspace(0.5, 0.95, 10)
    niou = iouv.shape[0]
    seen = 0
    jdict, stats, ap, ap_class = [], [], [], []

    for f in os.listdir(opt.dataset):
        _, ext = os.path.splitext(f)
        if ext != ".jpg" and ext != ".JPG" and ext != ".png" and ext != ".PNG":
            continue
        print(f)
        l = f.replace(f.split(".")[1], "txt")
        file_name = os.path.join(opt.dataset, f)
        label_name = os.path.join(opt.dataset.replace("images","labels"), l)
        try:
            labels = np.loadtxt(label_name)
        except:
            labels = np.array([])
        labels = labels.reshape((-1, 5))
        nl = labels.shape[0]

        p_u,p_l,p_d,p_r,img = make_squre(cv2.imread(file_name))
        if labels.shape[0]:
            labels[:,1] = labels[:,1]*(img.shape[1]-p_l-p_r)/img.shape[1] + p_l/img.shape[1]
            labels[:,2] = labels[:,2]*(img.shape[0]-p_u-p_d)/img.shape[0] + p_u/img.shape[0]
            labels[:,3] = labels[:,3]*(img.shape[1]-p_l-p_r)/img.shape[1]
            labels[:,4] = labels[:,4]*(img.shape[0]-p_u-p_d)/img.shape[0]

        img = cv2.resize(img, (opt.imgsz, opt.imgsz))
        labels[:,1:] = labels[:,1:]*opt.imgsz

        img = img.astype("float32")/255.
        img = nd.array(img.transpose((2,0,1))[None], ctx = ctx)

        nl = labels.shape[0]
        out = model(img).asnumpy()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
        pred = out[0]

        tcls = labels[:,0] if nl else []
        seen += 1
        
        if pred.shape[0] == 0:
            if nl:
                stats.append((np.zeros((0)), np.zeros((0)), np.zeros((0)), tcls))
            continue
        
        predn = pred.copy()
        scale_coords(img[0].shape[1:], predn[:, :4], [img.shape[2], img.shape[3]], [[1.0,1.0],[0.0,0.0]])  # native-space pred

        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img[0].shape[1:], tbox, [img.shape[2], img.shape[3]], [[1.0,1.0],[0.0,0.0]])  # native-space labels
            labelsn = np.concatenate((labels[:, 0:1], tbox), axis=1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = np.zeros((pred.shape[0], niou), dtype=np.bool)
        stats.append((correct, pred[:, 4], pred[:, 5], tcls))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=opt.classes)  # number of targets per class
    else:
        nt = np.zeros(1)

    s  = f'{opt.dataset}: #imges={seen}, #objects={nt.sum()}, mp={mp*100:02.2f}%, mr={mr*100:02.2f}%, map50={map50*100:02.2f}%, map={map*100:02.2f}%'
    fp = open(os.path.join("./results", opt.model+"_eval_float.txt"),"w")
    fp.write(s)
    fp.close()

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
