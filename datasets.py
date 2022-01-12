import os
from mxnet import gluon
import random
import cv2
import numpy as np
import math


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes 



def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        x = clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


class dataset(gluon.data.Dataset):
    def __init__(self,path='./dataset/train',classes=80,img_sizes=640,shuffle=True):
        super(dataset, self).__init__()
        self.img_size=img_sizes
        self.img_files = []
        self.lbl_files = []
        for f in os.listdir(os.path.join(path,"images")):
            if not os.path.isfile(os.path.join(path,"images",f)):
                continue
            self.img_files.append(os.path.join(path, "images", f))
            self.lbl_files.append(os.path.join(path, "labels", f.split(".")[0]+".txt"))
        self.len = len(self.img_files)
        self.images = []
        self.labels = []
        for i in range(self.len):
            if i % 10000 == 0:
                print("loading {} labels".format(i))
            lbl_name = self.lbl_files[i]
            if os.path.exists(lbl_name):
                lbl_data = np.loadtxt(lbl_name)
            else:
                lbl_data = np.zeros((0,5),dtype="float64")
            
            if len(lbl_data.shape) == 1:
                lbl_data = np.zeros((0,5),dtype="float64")
            self.labels.append(lbl_data)

        #self.batch = batch_size
        self.shape = (img_sizes, img_sizes)
        self.classes = classes
        self.mosaic_border = [-img_sizes//2, -img_sizes//2]
        self.path = path
        self.shuffle = shuffle
        self.classes = np.concatenate(self.labels, axis=0)[:,0]
        bincount = np.bincount(self.classes.astype("int32"), minlength=classes) + 10 
        bincount = np.sqrt(np.bincount(self.classes.astype("int32"), minlength=classes) + 10)
        self.weight = 1./bincount
        self.weight = self.weight * classes / np.sum(self.weight) 
        self.weight = np.ones((80), dtype="float32")
    def __len__(self):
        return len(self.img_files)

    def load_img(self, i):
        img = cv2.imread(self.img_files[i])
        h0, w0, _  = img.shape
        r = self.img_size/max(h0,w0)
        if r != 1:
            img = cv2.resize(img, (int(w0*r), int(h0*r)), interpolation=cv2.INTER_CUBIC)
        return img, (h0, w0), img.shape[:2]
        
    def __getitem__(self, index):
        yc, xc = (int(random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border)
        indices = [index]+[random.choice(range(self.len)) for _ in range(3)]
        random.shuffle(indices)
        labels4 = []
        for i, index in enumerate(indices):
            #img, _, (h,w) = self.images[index]
            img, _, (h,w) = self.load_img(index)
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((self.img_size * 2, self.img_size * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.labels[index].copy()
            if labels.size > 0:
                labels[:,1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            else:
                xx = 0
            labels4.append(labels)
        
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        labels4[:, 1:] = np.clip(labels4[:, 1:], 0, 2 * self.img_size)  # clip when using random_perspective()
        '''
        img = cv2.imread(self.img_files[index])
        h, w = img.shape[0:2]
        scale = min(1280/h, 1280/w)
        img4 = np.full((self.img_size * 2, self.img_size * 2, img.shape[2]), 114, dtype=np.uint8)
        img_large = cv2.resize(img, (int(scale*w), int(scale*h)))
        pad_w, pad_h = (1280 - img_large.shape[1])//2, (1280 - img_large.shape[0])//2
        img4[pad_h:pad_h+img_large.shape[0], pad_w:pad_w+img_large.shape[1], :] = img_large

        labels = self.labels[index].copy()
        labels4 = np.copy(labels)
        if labels.size > 0:
            #labels4[:,1:] = xywhn2xyxy(labels4[:, 1:], w, h, pad_w, pad_h)  # normalized xywh to pixel xyxy format
            labels4[:, 1] = w * scale * (labels[:, 1] - labels[:, 3] / 2) + pad_w  # top left x
            labels4[:, 2] = h * scale * (labels[:, 2] - labels[:, 4] / 2) + pad_h  # top left y
            labels4[:, 3] = w * scale * (labels[:, 1] + labels[:, 3] / 2) + pad_w  # bottom right x
            labels4[:, 4] = h * scale * (labels[:, 2] + labels[:, 4] / 2) + pad_h  # bottom right y
        '''
        '''
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']

        img0 = img4.copy()
        targets0 = labels4
        labels0 = targets0
        for i_label, label in enumerate(labels0):
            cls, left, top, right, bottom = label.astype("int32")
            #x_c, y_c, w, h = np.array([x_c, y_c, w, h])
            #left, top, right, bottom = int(x_c - w/2), int(y_c - h/2), int(x_c + w/2), int(y_c + h/2)
            img0 = cv2.rectangle(img0, (left, top), (right, bottom), (255,0,0))
            img0 = cv2.putText(img0, names[int(cls)], (left+4, top+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imshow("", img0)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        '''
        
        img, labels = random_perspective(img4, labels4, 
                                           degrees=0.0,
                                           translate=0.1,
                                           scale=0.5,
                                           shear=0.0,
                                           perspective=0.0,
                                           border=self.mosaic_border)
        
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        # Albumentations None
        # HSV color-space
        augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]
    
        labels_out = np.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = np.array(labels)

        # Convert
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        out = {"img": img, "label":labels_out}

        return img, labels_out
        
