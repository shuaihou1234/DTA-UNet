from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from Dynamic_convolution import *




##########消融实验###############
#使用Dynamic_convolution改变UNet下采样中的第二个卷积；也就是通道数目相同的卷积




#改动1动态卷积
def conv3x3(ch_in, ch_out, stride=1, groups=1,dilation=1):
    return Dynamic_conv2d(ch_in, ch_out, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False,dilation=dilation)



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            #conv3x3(ch_in, ch_out, stride=1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            conv3x3(ch_out, ch_out,stride=1),
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


class Dynamic_Convolution_AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Dynamic_Convolution_AttU_Net, self).__init__()

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
        self.sigmoid = nn.Sigmoid()

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
        d1 = self.sigmoid(d1)

        return d1

