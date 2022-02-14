import sys
from pathlib import Path
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon import HybridBlock


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


class SiLU(HybridBlock):
    def __init__(self, act):
        super().__init__();
        self.act = act
        #self.act = mx.ndarray.Activation(act_type='sigmoid')
    def hybrid_forward(self, F, x):
        if self.act == "silu":
            return x*nn.Activation(activation='sigmoid')(x)
        elif self.act == "relu":
            return nn.Activation(activation='relu')(x) # OK
        else:
            print("error in activation!")

class conv(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0, kernel=1, strid=1, padding=0, group=1, bais=False, act=None, fuse=False):
        super(conv, self).__init__()
        self.fuse = fuse
        self.conv = nn.Conv2D(channels=ch_out,kernel_size=(kernel,kernel),strides=(strid,strid), padding=padding, groups=group, use_bias=True if self.fuse else bais, activation=None)
        self.batch= None if self.fuse else nn.BatchNorm()
        self.silu = SiLU(act)
    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if not self.fuse:
            x = self.batch(x)
        x = self.silu(x)
        return x

class bottle(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(bottle, self).__init__()
        c_=int(ch_out*e)
        self.fuse = fuse
        self.conv1=conv(ch_in, c_, act=act, fuse=self.fuse)
        self.conv2=conv(c_,    ch_out, 3,1,1, act=act, fuse=self.fuse)
        self.add = short_cut if ch_in == ch_out else False
    def hybrid_forward(self, F, x):
        if self.add:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))        

class c3(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None):
        super(c3, self).__init__()
        c_ = int(ch_out*e)
        self.conv1 = conv(ch_in,c_)
        self.conv2 = conv(ch_in,c_)
        self.conv3 = conv(2*c_,ch_out)
        self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0, act=act) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        for bottle_i in self.bottle_list:
            m_out = bottle_i(m_out)
        return self.conv3(nd.concat(m_out, self.conv2(x), dim=1))

class c3_rep1(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep1, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep2(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep2, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep3(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep3, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep4(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep4, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle3 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        m_out = self.bottle3(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep6(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep6, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle3 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle4 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle5 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        m_out = self.bottle3(m_out)
        m_out = self.bottle4(m_out)
        m_out = self.bottle5(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep8(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep8, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle3 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle4 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle5 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle6 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle7 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        m_out = self.bottle3(m_out)
        m_out = self.bottle4(m_out)
        m_out = self.bottle5(m_out)
        m_out = self.bottle6(m_out)
        m_out = self.bottle7(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep9(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep9, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle3 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle4 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle5 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle6 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle7 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle8 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        m_out = self.bottle3(m_out)
        m_out = self.bottle4(m_out)
        m_out = self.bottle5(m_out)
        m_out = self.bottle6(m_out)
        m_out = self.bottle7(m_out)
        m_out = self.bottle8(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class c3_rep12(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0,repeat=1, short_cut=True, group=1, e=0.5, act=None, fuse=False):
        super(c3_rep12, self).__init__()
        c_ = int(ch_out*e)
        self.fuse = fuse
        self.conv1 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv2 = conv(ch_in,c_, act=act, fuse=self.fuse)
        self.conv3 = conv(2*c_,ch_out, act=act, fuse=self.fuse)
        self.bottle0 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle1 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle2 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle3 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle4 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle5 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle6 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle7 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle8 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle9 = bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle10= bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        self.bottle11= bottle(c_, c_, short_cut, group, 1.0, act=act, fuse=self.fuse)
        #self.bottle_list = [bottle(c_, c_, short_cut, group, 1.0) for _ in range(repeat)]
    def hybrid_forward(self, F, x):
        m_out = self.conv1(x)
        m_out = self.bottle0(m_out)
        m_out = self.bottle1(m_out)
        m_out = self.bottle2(m_out)
        m_out = self.bottle3(m_out)
        m_out = self.bottle4(m_out)
        m_out = self.bottle5(m_out)
        m_out = self.bottle6(m_out)
        m_out = self.bottle7(m_out)
        m_out = self.bottle8(m_out)
        m_out = self.bottle9(m_out)
        m_out = self.bottle10(m_out)
        m_out = self.bottle11(m_out)
        #for bottle_i in self.bottle_list:
        #    m_out = bottle_i(m_out)
        return self.conv3(F.concat(m_out, self.conv2(x), dim=1))

class sppf(HybridBlock):
    def __init__(self, ch_in=0, ch_out=0, k=5, act=None, fuse=False):
        super(sppf, self).__init__()
        c_ = ch_in//2
        self.fuse = fuse
        self.conv1=conv(ch_in, c_, act=act, fuse=self.fuse)
        self.conv2=conv(c_*4, ch_out, act=act, fuse=self.fuse)
        self.m = nn.MaxPool2D(pool_size=(k,k), strides=(1,1), padding=(k//2, k//2))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(F.concat(x, y1, y2, y3, dim=1))

class resize(HybridBlock):
    def __init__(self):
        super(resize, self).__init__()
    def hybrid_forward(self, F, x):
        #return nd.contrib.BilinearResize2D (x, scale_height=2.0, scale_width=2.0)
        x = F.repeat(x, 2, axis=2)
        x = F.repeat(x, 2, axis=3)
        return x

class cat(HybridBlock):
    def __init__(self, dim=1):
        super(cat, self).__init__()
        self.dim = dim
    def hybrid_forward(self, F, x, y):
        return F.concat(x, y, dim=self.dim)

class detect(HybridBlock):
    def __init__(self, batch_size=1, nc=80, anchors=(),ch=(),inplace=True, mode="train", ctx=mx.cpu()):
        super(detect, self).__init__()
        self.ctx = ctx
        self.mode = mode
        self.batch_size = batch_size
        self.nc=nc
        self.no=nc+5
        self.nl=len(anchors)
        self.na=len(anchors[0])//2
        self.grid=[nd.zeros((1), ctx=self.ctx)]*self.nl
        self.anchor_grid=[nd.zeros((1), ctx=self.ctx)]*self.nl
        self.stride = nd.array([8., 16., 32.], ctx=self.ctx)
        self.m0 = nn.Conv2D(self.no*self.na,1)
        self.m1 = nn.Conv2D(self.no*self.na,1)
        self.m2 = nn.Conv2D(self.no*self.na,1)
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
    def hybrid_forward(self, F, x, y, z):
        #for i in range(self.nl):
        #    x[i] = self.m[i](x[i])
        #    bs, _, ny, nx = x[i].shape
        #    x[i] = nd.transpose(x[i].reshape(bs, self.na, self.no, ny, nx), axes=(0,1,3,4,2))
        x = self.m0(x)
        y = self.m1(y)
        z = self.m2(z)
        if self.mode == "train":
            #bs, _, ny, nx = x[0].shape
            bs, _, ny, nx = x.infer_shape(data=(self.batch_size,3,640,640))[1][0]
            #x[0] = F.transpose(x[0].reshape(bs, self.na, self.no, ny, nx), axes=(0,1,3,4,2))
            x = F.transpose(x.reshape((bs, self.na, self.no, ny, nx)), axes=(0,1,3,4,2))

            #bs, _, ny, nx = x[1].shape
            bs, _, ny, nx = y.infer_shape(data=(self.batch_size,3,640,640))[1][0]
            #x[1] = F.transpose(x[1].reshape(bs, self.na, self.no, ny, nx), axes=(0,1,3,4,2))
            y = F.transpose(y.reshape((bs, self.na, self.no, ny, nx)), axes=(0,1,3,4,2))

            #bs, _, ny, nx = x[2].shape
            bs, _, ny, nx = z.infer_shape(data=(self.batch_size,3,640,640))[1][0]
            #x[2] = F.transpose(x[2].reshape(bs, self.na, self.no, ny, nx), axes=(0,1,3,4,2))
            z = F.transpose(z.reshape((bs, self.na, self.no, ny, nx)), axes=(0,1,3,4,2))
            #if not self.training:
            return (x, y, z)
        else:
            out = []
            bs, _, ny, nx = x.shape
            x = x.reshape((bs, self.na, self.no, ny, nx)).transpose((0,1,3,4,2))
            self.grid[0], self.anchor_grid[0] = self._make_grid(nx, ny, 0)
            tmp = x.sigmoid()
            xy = (tmp[..., 0:2] * 2 - 0.5 + self.grid[0]) * self.stride[0]  # xy
            wh = (tmp[..., 2:4] * 2) ** 2 * self.anchor_grid[0]  # wh
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            bs, _, ny, nx = y.shape
            y = y.reshape((bs, self.na, self.no, ny, nx)).transpose((0,1,3,4,2))
            self.grid[1], self.anchor_grid[1] = self._make_grid(nx, ny, 1)
            tmp = y.sigmoid()
            xy = (tmp[..., 0:2] * 2 - 0.5 + self.grid[1]) * self.stride[1]  # xy
            wh = (tmp[..., 2:4] * 2) ** 2 * self.anchor_grid[1]  # wh
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            bs, _, ny, nx = z.shape
            z = z.reshape((bs, self.na, self.no, ny, nx)).transpose((0,1,3,4,2))
            self.grid[2], self.anchor_grid[2] = self._make_grid(nx, ny, 2)
            tmp = z.sigmoid()
            xy = (tmp[..., 0:2] * 2 - 0.5 + self.grid[2]) * self.stride[2]  # xy
            wh = (tmp[..., 2:4] * 2) ** 2 * self.anchor_grid[2]  # wh
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            return nd.concat(*out, dim=1)

    def _make_grid(self, nx=20, ny=20, i=0):
        yv = nd.array(range(ny))[:,None].repeat(nx,axis=1)
        xv = nd.array(range(nx))[None,:].repeat(ny,axis=0)
        grid = nd.concat(xv[...,None], yv[...,None], dim=2)[None,None,...].repeat(self.na, axis=1)
        grid = nd.Cast(grid, dtype="float32")

        anchor_grid = (self.anchors[i].copy() * self.stride[i])
        anchor_grid = anchor_grid[None,:, None, None,:]
        anchor_grid = anchor_grid.repeat(ny, axis=-3)
        anchor_grid = anchor_grid.repeat(nx, axis=-2)
        return grid, anchor_grid


class yolov5(HybridBlock):
    def __init__(self, batch_size = 16, classes=80, mode="train", ctx=mx.cpu(), act="silu", gd=1, gw=1, fuse=False):
        super(yolov5, self).__init__()
        self.ctx = ctx
        self.mode = mode
        self.batch_size = batch_size
        self.classes = classes
        self.act = act
        self.fuse = fuse
        self.conv1 = conv(3, 16*gw,6,2,2, act=self.act, fuse=self.fuse)
        self.conv2 = conv(16*gw,32*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.c3_1 = eval(f'c3_rep{gd*1}')(32*gw,32*gw,1,True,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv3 = conv(32*gw,64*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.c3_2 = eval(f'c3_rep{gd*2}')(64*gw,64*gw,2,True,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv4 = conv(64*gw,128*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.c3_3 = eval(f'c3_rep{gd*3}')(128*gw,128*gw,3,True,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv5 = conv(128*gw,256*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.c3_4 = eval(f'c3_rep{gd*1}')(256*gw,256*gw,1,True,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.sppf = sppf(256*gw,256*gw,5, act=self.act, fuse=self.fuse)
        self.conv6 = conv(256*gw,128*gw,1,1, act=self.act, fuse=self.fuse)
        self.upsample1 = resize()
        self.cat1  = cat(dim=1)
        self.c3_5 = eval(f'c3_rep{gd*1}')(256*gw,128*gw,1,False,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv7 = conv(128*gw,64*gw,1,1, act=self.act, fuse=self.fuse)
        self.upsample2 = resize()
        self.cat2  = cat(dim=1)
        self.c3_6 = eval(f'c3_rep{gd*1}')(128*gw,64*gw,1,False,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv8 = conv(64*gw,64*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.cat3  = cat(dim=1)
        self.c3_7 = eval(f'c3_rep{gd*1}')(128*gw,128*gw,1,False,group=1,e=0.5, act=self.act, fuse=self.fuse)
        self.conv9 = conv(128*gw,128*gw,3,2,1, act=self.act, fuse=self.fuse)
        self.cat4  = cat(dim=1)
        self.c3_8 = eval(f'c3_rep{gd*1}')(256*gw,256*gw,1,False,group=1,e=0.5, act=self.act, fuse=self.fuse)
        anchors = [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]
        self.det  = detect(self.batch_size, nc=self.classes, anchors=anchors,ch=[64*gw,128*gw,256*gw],inplace=True, mode=self.mode,ctx=self.ctx)
    def hybrid_forward(self, F, x):
        x = self.conv1(x)              #0
        x = self.conv2(x)              #1
        x = self.c3_1(x)               #2
        x = self.conv3(x)              #3
        c3_2 = self.c3_2(x)            #4
        x = self.conv4(c3_2)           #5
        c3_3 = self.c3_3(x)            #6
        x = self.conv5(c3_3)           #7
        x = self.c3_4(x)               #8
        x = self.sppf(x)               #9
        conv6 = self.conv6(x)          #10
        x = self.upsample1(conv6)      #11
        x = self.cat1(x,c3_3)          #12
        x = self.c3_5(x)               #13
        conv7 = self.conv7(x)          #14
        x = self.upsample2(conv7)      #15
        x = self.cat2(x,c3_2)          #16
        c3_6 = self.c3_6(x)            #17
        x = self.conv8(c3_6)           #18
        x = self.cat3(x,conv7)         #19
        c3_7 = self.c3_7(x)            #20
        x = self.conv9(c3_7)           #21
        x = self.cat4(x,conv6)         #22
        c3_8 = self.c3_8(x)            #23
        out = self.det(c3_6,c3_7,c3_8) #24
        return out
