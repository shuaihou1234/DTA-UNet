from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import math
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class DYCls(nn.Module):
    def __init__(self, inp, oup):
        super(DYCls, self).__init__()
        self.dim = 32
        self.cls = nn.Linear(inp, oup)
        self.cls_q = nn.Linear(inp, self.dim, bias=False)
        self.cls_p = nn.Linear(self.dim, oup, bias=False)

        mid = 32

        self.fc = nn.Sequential(
            nn.Linear(inp, mid, bias=False),
            SEModule_small(mid),
        )
        self.fc_phi = nn.Linear(mid, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(mid, oup, bias=False)
        self.hs = Hsigmoid()
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

    def forward(self, x):
        # r = self.cls(x)
        b, c = x.size()
        y = self.fc(x)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1)

        r = dy_scale * self.cls(x)

        x = self.cls_q(x)
        x = self.bn1(x)
        x = self.bn2(torch.matmul(dy_phi, x.view(b, self.dim, 1)).view(b, self.dim)) + x
        x = self.cls_p(x)

        return x + r




class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = scale.expand_as(r) * r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b, -1, h, w)
        out = self.p(out) + r
        return out


class Bottleneck_dy(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_dy, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv_dy(inplanes, planes, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_dy(planes, planes, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_dy(planes, planes * 4, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            conv_dy(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

'''size – 根据不同的输入类型制定的输出大小

scale_factor – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型

mode (str, optional) – 可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. 默认使用'nearest'

align_corners (bool, optional) – 如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。'''

#padding 的操作就是在图像块的周围加上格子, 从而使得图像经过卷积过后大小不会变化,这种操作是使得图像的边缘数据也能被利用到,这样才能更好地扩张整张图像的边缘特征.


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # 上采样的 l 卷积
        x1 = self.W_x(x)                  #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        # concat + relu
        psi = self.relu(g1 + x1)          #1x256x64x64di
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)               #得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）
        # 返回加权的 x
        return x * psi                    #与low-level feature相乘，将权重矩阵赋值进去


class DCD_AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(DCD_AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)               ## 1*3*512*512 ->conv(3,64)->conv(64,64)-> 1*64*512*512

        x2 = self.Maxpool(x1)            # 1*64*512*512 -> 1*64*256*256
        x2 = self.Conv2(x2)              # 1*64*256*256 ->conv(64,128)->conv(128,128)-> 1*128*256*256

        x3 = self.Maxpool(x2)            # 1*128*256*256 -> 1*128*128*128
        x3 = self.Conv3(x3)              ## 1*128*128*128 ->conv(128,256)->conv(256,256)->  1*256*128*128

        x4 = self.Maxpool(x3)            # 1*256*128*128 -> 1*256*64*64
        x4 = self.Conv4(x4)              ## 1*256*64*64 ->conv(256,512)->conv(512,512)-> 1*512*64*64

        x5 = self.Maxpool(x4)            ## 1*512*64*64 -> 1*512*32*32
        x5 = self.Conv5(x5)             ## 1*512*32*32->conv(512,1024)->conv(1024,1024)-> 1*1024*32*32

        # decoding + concat path
        d5 = self.Up5(x5)                ## 1*1024*32*32 ->Upsample-> 1*1024*64*64 -> conv(1024,512) ->1*512*64*64
        x4 = self.Att5(g=d5, x=x4)        ## 2(1*512*64*64) -> 1*1*64*64 ->1*512*64*64
        d5 = torch.cat((x4, d5), dim=1)    ## 1*1024*64*64
        d5 = self.Up_conv5(d5)              ## 1*1024*64*64 ->conv(1024,512)->conv(512,512)-> 1*512*64*64

        d4 = self.Up4(d5)                   #1*512*64*64->Upsample-> 1*512*128*128 -> conv(512,256) ->1*256*128*128
        x3 = self.Att4(g=d4, x=x3)          ## 2(1*256*128*128) -> 1*1*128*128 ->1*256*128*128
        d4 = torch.cat((x3, d4), dim=1)     ## 1*512*128*128
        d4 = self.Up_conv4(d4)              ## 1*512*128*128 ->conv(512,256) -> conv(512,512)-> 1*256*128*128

        d3 = self.Up3(d4)                   #1*256*128*128->Upsample-> 1*256*256*256 -> conv(256,128) ->1*128*256*256
        x2 = self.Att3(g=d3, x=x2)          #2(1*128*256*256) -> 1*1*256*256 -> 1*128*256*256

        d3 = torch.cat((x2, d3), dim=1)       #1*256*256*256
        d3 = self.Up_conv3(d3)                #1*256*256*256->conv(256,128) -> conv(128,128)-> 1*128*256*256

        d2 = self.Up2(d3)                      #1*128*256*256->Upsample-> 1*128*512*512 -> conv(128,64) ->1*64*512*512
        x1 = self.Att2(g=d2, x=x1)              #2(1*64*512*512)  -> 1*1*512*512  ->1*64*512*512
        d2 = torch.cat((x1, d2), dim=1)         #1*128*512*512
        d2 = self.Up_conv2(d2)                  #1*128*512*512 -> conv(128,64) ->1*64*512*512 -> conv(64,64) ->1*64*512*512

        d1 = self.Conv_1x1(d2)                  #1*64*512*512 -> 1*3*512*512
        d1 = self.softmax(d1,1)

        return d1

