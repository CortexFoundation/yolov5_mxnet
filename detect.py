#!/usr/bin/python
#coding:utf-8

import argparse
import os
import sys
from datasets import dataset
#from datasets import xywh2xyxy

import math
import numpy as np
from mxnet import nd
from yolo import yolov5
import cv2
import mxnet as mx
import logging
import re
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import platform

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows
def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path
CONFIG_DIR = user_config_dir()

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)

def check_font(font='Arial.ttf', size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except:
        pass

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)




        
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    def clip_coords(boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords





def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    def NMS(dets, scores, thresh):
        # x1, y1, x2, y2, score
        # （x1、y1）（x2、y2）left-top, and right-bottom
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        #area for each candidate
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        #descending reorder
        order = scores.argsort()[::-1]

        temp = []
        while order.size > 0:
            i = order[0]
            temp.append(i)
            # calc the interact coordinate between current largest prob. candidate and others
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.minimum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.maximum(y2[i], y2[order[1:]])

            # calc the area of interact condidate, note the area should be nonzero
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # find the candidate with interactness larger than threshold
            inds = np.where(ovr <= thresh)[0]
            #re-order
            order = order[inds + 1]
        return np.array(temp)

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            j, i = np.where(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype("float32")), axis=1)
        else:  # best class only
            j = np.argmax(x[:, 5:], axis=1)
            conf = np.array([x[i,5+j[i]] for i in range(x.shape[0])])
            x = np.concatenate((box, conf[:, None], j[:, None].astype("float32")), axis=1)[conf.reshape((-1)) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = NMS(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",         type=bool,  default=True,     help="whether or not using cpu for training")
    parser.add_argument("--gpu",         type=int,   default=2,         help="which gpu used for training")
    parser.add_argument("--batch_size",  type=int,   default=1,         help="batch size used for training")
    parser.add_argument("--classes",     type=int,   default=80,        help="how many classes for the detection and classfication problem")
    parser.add_argument("--imgsz",       type=int,   default=640,       help="input image size")
    parser.add_argument("--dataset",     type=str,   default="./dataset/trial/images",   help="trial data for debug or training")
    parser.add_argument("--model_dir",   type=str,   default="./",      help="Model dir for save and load")
    parser.add_argument("--model",       type=str,   default="yolov5l", help="model name")
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
        NAME = []
        EPOCH = []
        start_epoch = 0
        for f in os.listdir(os.path.join("./weights",args.model_dir)):
            if f.endswith("params"):
                name_epoch = f.strip().split(".")[0].split("-")
                if len(name_epoch) == 2 and name_epoch[0] == args.model:
                    EPOCH.append(name_epoch[1])
        tmp = [int(_) for _ in EPOCH]
        ind = tmp.index(max(tmp))
        params_file = os.path.join("./weights", args.model_dir, args.model+"-"+EPOCH[ind]+".params")
        model.collect_params().load(params_file,ignore_extra=False)
        
        print(f'load weight {params_file} successfully')
    except:
        print("failed to load weight")

    dirs = os.path.join("./results/", opt.model) 
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
