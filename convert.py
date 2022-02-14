#!/usr/bin/python
#coding:utf-8

import argparse
import os
from mxnet import nd
from yolo import yolov5
import mxnet as mx
from utils import from_torch_model, str2bool


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=str2bool,default=False,   help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,   default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,   default=1,         help="batch size used for training")
    parser.add_argument("--model_dir",   type=str,   default="./weights/",  help="Model dir for save and load")
    parser.add_argument("--model",       type=str,   default="yolov5m", help="model name")
    parser.add_argument("--fuse",        type=str,   default=True,      help="activation with silu or relu")
    opt = parser.parse_args()
    return opt


def main(opt):
    ctx = mx.cpu() if opt.cpu else mx.gpu(opt.gpu)
    print(opt)
    print(ctx)

    gw = {"n":1, "s":2, "m":3, "l":4, "x":5}
    gd = {"n":1, "s":1, "m":2, "l":3, "x":4}
    postfix = opt.model[-1]
    model = yolov5(batch_size=opt.batch_size, mode="train", ctx=ctx, act=opt.silu, gd=gd[postfix], gw=gw[postfix], fuse=True)
    model.collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
    model.hybridize()

    model(nd.zeros((1,3,640,640), ctx=ctx))

    from_torch_model(os.path.join(opt.model_dir, opt.model+".npy"), model, ctx)

    model.export(os.path.join("./weights",opt.model),epoch=9999)
    print("end")
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
