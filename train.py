#!/usr/bin/python
#coding:utf-8

import os
import mxnet as mx
import argparse
import numpy as np
from mxnet import nd
from yolo import yolov5
from datasets import dataset
from mxnet import gluon
from loss import ComputeLoss
from mxnet import autograd
from loss import build_targets
import logging
import cv2
from utils import str2bool

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=str2bool,  default=False,     help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,   default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,   default=4,        help="batch size")
    parser.add_argument("--imgsz",       type=int,   default=640,       help="input image size")
    parser.add_argument("--lr",          type=float, default=0.001,      help="learning rate")
    parser.add_argument("--classes",     type=int,   default=80,        help="how many classes for the detection and classfication problem")
    parser.add_argument("--dataset",     type=str,   default="./dataset/trial",   help="trial data for debug or training")
    parser.add_argument("--model_dir",   type=str,   default="./weights",      help="Model dir for save and load")
    parser.add_argument("--model",       type=str,   default="yolov5x", help="model name")
    parser.add_argument("--fuse",        type=str2bool,   default=True,    help="fuse conv and normal")
    parser.add_argument("--step",        type=list,  default=[300000, 600000], help="period for lr update when training")
    opt = parser.parse_args()
    return opt

def main(opt):
    args = parse_opt()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = args.model+"_log.log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ctx = mx.cpu() if args.cpu else mx.gpu(args.gpu)

    def batchify_fn(x):
        img, lbl = [], []
        for i, x0 in enumerate(x):
            img.append(x0[0][np.newaxis,:])
            label = x0[1]
            label[:,0] = i
            lbl.append(label)
        img = np.concatenate(img, axis=0)
        lbl = np.concatenate(lbl, axis=0)
        return (img, lbl)
    
    train_dataset = dataset(path=args.dataset,classes=args.classes,img_sizes=args.imgsz,shuffle=True)
    train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, batchify_fn=batchify_fn, last_batch='discard')
    final_loss = ComputeLoss(ctx=ctx, pos_weight=nd.array(train_dataset.weight, ctx=ctx))

    gw = {"n":1, "s":2, "m":3, "l":4, "x":5}
    gd = {"n":1, "s":1, "m":2, "l":3, "x":4}
    postfix = args.model[-1]
    model = yolov5(batch_size=args.batch_size, mode="train", ctx=ctx, gd=gd[postfix], gw=gw[postfix], fuse=args.fuse)

    model.collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
    model.hybridize()

    try:
        NAME = []
        EPOCH = []
        for f in os.listdir(args.model_dir):
            if f.endswith("params"):
                name_epoch = f.strip().split(".")[0].split("-")
                if len(name_epoch) == 2 and name_epoch[0] == args.model:
                    EPOCH.append(name_epoch[1])
        tmp = [int(_) for _ in EPOCH]
        ind = tmp.index(max(tmp))
        params_file = os.path.join(args.model_dir, args.model+"-"+EPOCH[ind]+".params")
        model.collect_params().load(params_file,ignore_extra=False)
        print("load weight successfully")
    except:
        print("failed to load weight")

    steps = [int(x*16/args.batch_size) for x in args.step]

    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.5)
    optimizer = mx.optimizer.Adam(learning_rate=args.lr, lr_scheduler=schedule)
    trainer = gluon.Trainer(model.collect_params(), optimizer=optimizer)

    iter = 0
    while True:
        for imgs, labels in train_dataloader:
            imgs = nd.array(imgs.astype("float32")/255.).as_in_context(ctx)

            with autograd.record():
                pred = model(imgs)
            tcls, tbox, indices, anchors = build_targets(pred, labels, ctx=ctx, imgsize=args.imgsz)
            
            # if there is no object in this batch, just skip gradient update and jump to next batch
            if indices[0][0].shape[0] == 0:
                continue
            with autograd.record():
                loss, lbox, lobj, lcls = final_loss(pred, tcls, tbox, indices, anchors)
            loss.backward()
            trainer.step(1, ignore_stale_grad=True)
            
            with autograd.pause():
                lbox_np = lbox.asscalar()
                lobj_np = lobj.asscalar()
                lcls_np = lcls.asscalar()
            txt = "iter {:6d}: loss = {:4f}, lbox = {:4f}, lobj = {:4f}, lcls = {:4f}, bs={:2d}, lr={:5f}".format(iter, lbox_np+lobj_np+lcls_np, lbox_np, lobj_np, lcls_np, args.batch_size, schedule.base_lr)           
            print(txt)
            logger.info(txt)
            if iter % 20000 == 19999:
                model.export(os.path.join("./weights",args.model), epoch=int(iter/len(train_dataset)*args.batch_size))

            iter = iter + 1

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
