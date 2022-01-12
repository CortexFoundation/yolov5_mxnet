import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from mxnet import autograd
from mxnet.gluon import Block
from mxnet import nd

def build_targets(pred, targets, ctx, imgsize=640):
    ANCHORS = np.array([[[ 1.25000,  1.62500],
                         [ 2.00000,  3.75000],
                         [ 4.12500,  2.87500]],
                         [[ 1.87500,  3.81250],
                         [ 3.87500,  2.81250],
                         [ 3.68750,  7.43750]],
                         [[ 3.62500,  2.81250],
                         [ 4.87500,  6.18750],
                         [11.65625, 10.18750]]])
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 3, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = np.ones(7)  # normalized to gridspace gain
    ai = np.arange(na, dtype="float32").reshape((na, 1)).repeat(nt, axis=1)  # same as .repeat_interleave(nt)
    targets = np.concatenate((targets[None].repeat(na,0), ai[...,None]), axis=2)  # append anchor indices

    g = 0.5  # bias
    off = np.array([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], dtype="float32") * g  # offsets

    for i in range(3):
        anchors = ANCHORS[i]
        gain[2:6] = np.array(pred[i].shape, dtype="int32")[[3, 2, 3, 2]]  # xyxy gain
        #gain[2:6] = np.ones((4), dtype="float32")*640/8*np.power(2, i)
        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = np.maximum(r, 1/r).max(2) < 4.0  # compare
            t = t[j]

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1 < g) * (gxy > 1)).transpose()
            l, m = ((gxi % 1 < g) * (gxi > 1)).transpose()
            j = np.stack((np.ones_like(j), j, k, l, m))
            t = t[None].repeat(5, axis=0)[j]
            offsets = (np.zeros_like(gxy)[None]+off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].astype("int32").T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).astype("int32")
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].astype("int32") # anchor indices
        indices.append((nd.array(b, ctx=ctx),nd.array(a, ctx=ctx),nd.array(gj.clip(0, gain[3]-1), ctx=ctx), nd.array(gi.clip(0, gain[2]-1), ctx=ctx)))
        tbox.append(nd.array(np.concatenate((gxy-gij,gwh), axis=1), ctx=ctx))  # box
        anch.append(nd.array(anchors[a], ctx=ctx))  # anchors
        #tcls.append(nd.array(c, ctx=ctx))  # class
        n = c.shape[0]
        t = np.zeros((n, 80), dtype="float32")
        t[range(n), c] = 1.
        tcls.append(nd.array(t, ctx=ctx))

    return tcls, tbox, indices, anch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = nd.maximum(nd.minimum(b1_x2, b2_x2) - nd.maximum(b1_x1, b2_x1), 0) * \
            nd.maximum(nd.minimum(b1_y2, b2_y2) - nd.maximum(b1_y1, b2_y1), 0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = nd.maximum(b1_x2, b2_x2) - nd.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = nd.maximum(b1_y2, b2_y2) - nd.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * nd.power(nd.arctan(w2 / h2) - nd.arctan(w1 / h1), 2)
                with autograd.pause():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class ComputeLoss(mx.gluon.loss.Loss):
    def __init__(self, ctx, pos_weight=None, **kwargs):
        super(ComputeLoss, self).__init__(weight=None, batch_axis=None,**kwargs)
        self.sort_obj_iou = False
        BCEcls = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=1.0)
        BCEobj = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=1.0)
        self.cp = 1.0
        self.cn = 0.0
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.balance = nd.array([4.0, 1.0, 0.4], ctx=ctx)
        self.ssi = 0
        self.gr = 1.0
        self.autobalance = False
        self.ctx = ctx
        self.pos_weight = pos_weight
        hyp = {
                'lr0': 0.01, 
                'lrf': 0.1, 
                'momentum': 0.937, 
                'weight_decay': 0.0005, 
                'warmup_epochs': 3.0, 
                'warmup_momentum': 0.8, 
                'warmup_bias_lr': 0.1, 
                'box': 0.05, 
                'cls': 0.5, 
                'cls_pw': 1.0, 
                'obj': 1.0, 
                'obj_pw': 1.0, 
                'iou_t': 0.2, 
                'anchor_t': 4.0, 
                'fl_gamma': 0.0, 
                'hsv_h': 0.015, 
                'hsv_s': 0.7, 
                'hsv_v': 0.4, 
                'degrees': 0.0, 
                'translate': 0.1, 
                'scale': 0.5, 
                'shear': 0.0, 
                'perspective': 0.0, 
                'flipud': 0.0, 
                'fliplr': 0.5, 
                'mosaic': 1.0, 
                'mixup': 0.0, 
                'copy_paste': 0.0, 
                'label_smoothing': 0.0
              }
        self.hyp = hyp
        self.na = 3
        self.nc = 80
        self.nl = 3
        self.anchors = nd.array([[[ 1.25000,  1.62500],
                                  [ 2.00000,  3.75000],
                                  [ 4.12500,  2.87500]],
                                 [[ 1.87500,  3.81250],
                                  [ 3.87500,  2.81250],
                                  [ 3.68750,  7.43750]],
                                 [[ 3.62500,  2.81250],
                                  [ 4.87500,  6.18750],
                                  [11.65625, 10.18750]]
                                 ], ctx=self.ctx)
        
    def forward(self, p, tcls, tbox, indices, anchors):
        lcls, lbox, lobj = nd.zeros(1, self.ctx), nd.zeros(1, self.ctx), nd.zeros(1, self.ctx)
        #tcls, tbox, indices, anchors = self.build_targets(p, targets) 
        #log = ""
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            #pi = pi.asnumpy()
            #print(indices[i][0].shape, indices[i][1].shape,indices[i][2].shape,indices[i][3].shape,)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            #tobj = nd.zeros_like(pi[:,:,:,:, 0])  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                #log = log + "n={:} ".format(n)                
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = nd.Concat(pxy, pwh, dim=1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()

                # Objectness
                score_iou = iou.detach()
                if self.sort_obj_iou:
                    sort_id = nd.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]

                if self.nc > 1: 
                    #t = np.zeros_like(ps[:, 5:])
                    #t[range(n), tcls[i]] = self.cp
                    lcls = lcls + self.BCEcls(ps[:, 5:], tcls[i], None, self.pos_weight).mean()
            else:
                continue
            
            EPS = 1e-10
            #log = log + "score_iou.shape[0]={:}".format(score_iou.shape[0])
            if score_iou.shape[0] > 0:
                ratio = 5.
                size_obj = b.size
                size_non = pi[...,4].size - size_obj
                #weight = nd.array([2*size_non/(size_non+ratio*size_obj), 2*ratio*size_obj/(size_non+ratio*size_obj)], ctx=self.ctx)
                obj0 = - nd.log(EPS+pi[b,a,gj,gi,4].sigmoid())*score_iou - nd.log(EPS+1-pi[b,a,gj,gi,4].sigmoid())*(1-score_iou)
                obj1 = - nd.log(EPS + 1 - pi[:,:,:,:,4].sigmoid())
                obj2 = - nd.log(EPS + 1 - pi[b,a,gj,gi,4].sigmoid())
                obji = obj0.sum()*2 + (obj1.sum() - obj2.sum())/10
                obji = obji/pi.shape[0]/pi.shape[1]/pi.shape[2]/pi.shape[3]
                lobj = lobj + obji * self.balance[i]  # obj loss
            else:
                lobj = 0.

            '''
            if score_iou.shape[0] > 0:
                all_zeros = nd.zeros_like(pi[:,:,:,:, 4])
                ind_zeros = nd.zeros_like(pi[b,a,gj,gi,4])
                obji = self.BCEobj(pi[b,a,gj,gi,4], score_iou).sum() + self.BCEobj(pi[:,:,:,:,4], all_zeros).sum() - self.BCEobj(pi[b,a,gj,gi,4], ind_zeros).sum()
                obji = obji/pi.shape[0]/pi.shape[1]/pi.shape[2]/pi.shape[3]
                lobj = lobj + obji * self.balance[i]  # obj loss
            else:
                lobj = 0.
            '''
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox = lbox * self.hyp['box'] # 0.05
        lobj = lobj * self.hyp['obj'] # 1
        lcls = lcls * self.hyp['cls'] # 0.5
        #print(log)
        return lbox + lobj + lcls, lbox, lobj, lcls
