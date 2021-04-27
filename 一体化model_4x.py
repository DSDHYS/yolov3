import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
import math
from collections import OrderedDict

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernal_size,stride,padding):
        super(ConvolutionalLayer,self).__init__()
        # self.out_channels=32# Conv2d 32x32x3
        # self.conv1=nn.Conv2d(3,self.out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        # #第一个3是因为inputs通道数为3;kernel_size=3是卷积核的通道数,32x32x3中的3;
        # self.BN1=nn.BatchNorm2d(self.out_channels)#参数为卷积后输入尺寸;该步进行归一化
        # self.relu1 = nn.LeakyReLU(0.1)#激活函数,一般取0.1
        # #实现inputs-Conv2D 32x3x3; 416,416,3 -> 416,416,32；第一个卷积
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):#前向传递
        return self.conv1(x)


    #残差网络,由一个1x1的卷积核一个3x3的卷积构成
class ResidualLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResidualLayer, self).__init__()        
        self.ReseBlock=nn.Sequential(
            ConvolutionalLayer(in_channels,in_channels//2,kernal_size=1,stride=1,padding=0),#第一次的输出变为输入的1/2，通道收缩#1X1的卷积核，所以padding=0
            ConvolutionalLayer(in_channels//2,in_channels,kernal_size=3,stride=1,padding=1)#第二次的输出变为输入的2倍，通道扩张还原，3x3的卷积核，416x416需要进行0填充
        )

    def forward(self,x):
        return x+self.ReseBlock(x)#X+两次卷积的结果，详见残差网络流程图

    #残差块的叠加,分别进行1，2，8，8，4次
# class make_layers(nn.Module):
#     def __init__(self,in_channels,count):
#         super(make_layers, self).__init__()
#         self.count= count
#         self.in_channels=in_channels
#         self.RB=ResidualLayer(self.in_channels)
#     def forward(self,x):#for循环实现叠加
#         for i in range(0,self.count):
#             x=self.RB(x)
#         return x

# #下采样层
# class DownSampleLayer(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(DownSampleLayer, self).__init__()
#         self.conv=ConvolutionalLayer(in_channels,out_channels,kernal_size=3, stride=2, padding=1)#见cfg文件
#     def forward(self,x):
#         return self.conv(x)

def make_layer(in_channels,out_channels, count):
    layers = []
    for i in range(0, count):
        layers.append(("residual_{}".format(i), ResidualLayer(out_channels)))
    return nn.Sequential(OrderedDict(layers))


#下采样层
class DownSampleLayer(nn.Module):
    def __init__(self,in_channels,out_channels,count):
        super(DownSampleLayer, self).__init__()
        self.DS=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,kernal_size=3, stride=2, padding=1),#见cfg文件
            make_layer(in_channels,out_channels,count),
        )
    def forward(self,x):
        return self.DS(x)
#上采样层
class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()
    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='nearest')

#搭建DarkNet53,获得三个特征层
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.RB_104=nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),#32X32X3
            DownSampleLayer(32, 64,1),#下采样，通道扩张
            #ResidualLayer(64),#残差网络
            DownSampleLayer(64, 128,2),
            #make_layers(128, 2),#进行两次残差

            #make_layers(256, 8)
        )
        self.RB_52=nn.Sequential(
            DownSampleLayer(128,256,8),
            #make_layers(512,8)
        )
        self.RB_26=nn.Sequential(
            DownSampleLayer(256,512,8),
            #make_layers(1024,4),
        )
        self.RB_13=nn.Sequential(
            DownSampleLayer(512,1024,4),
            #make_layers(1024,4),
        )
        self.contact_13=nn.Sequential(
            Conv2d_Block_5L(1024,512)
        )
        self.contact_26=nn.Sequential(
            Conv2d_Block_5L(768,256)
        )
        self.contact_52=nn.Sequential(
            Conv2d_Block_5L(384,128)
        )
        self.contact_104=nn.Sequential(
            Conv2d_Block_5L(192,64)
        )

        self.out_13=nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,33,1,1,0)#33=(5+检测类数)*3
        )
        self.out_26=nn.Sequential(
            ConvolutionalLayer(256,512,3,1,1),
            nn.Conv2d(512,33,1,1,0)#33=(5+检测类数)*3
        )
        self.out_52=nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256,33,1,1,0)#33=(5+检测类数)*3
        )
        self.out_104=nn.Sequential(
            ConvolutionalLayer(64,128,3,1,1),
            nn.Conv2d(128,33,1,1,0)#33=(5+检测类数)*3
        )

        self.up_104=nn.Sequential(
            ConvolutionalLayer(128,64,1,1,0),
            UpSampleLayer(),
            )#上采样,
        self.up_52=nn.Sequential(
            ConvolutionalLayer(256,128,1,1,0),
            UpSampleLayer()
            )#上采样,
        self.up_26=nn.Sequential(
            ConvolutionalLayer(512,256,1,1,0),
            UpSampleLayer(),
            )#上采样,

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self,x):
        RB_104=self.RB_104(x)
        RB_52=self.RB_52(RB_104)
        RB_26=self.RB_26(RB_52)
        RB_13=self.RB_13(RB_26)
        
        conval_13=self.contact_13(RB_13)
        out_13 = self.out_13(conval_13)

        up_26 = self.up_26(conval_13)
        route_26 = torch.cat((up_26,RB_26),dim=1)
        conval_26 = self.contact_26(route_26)
        out_26= self.out_26(conval_26)

        up_52 = self.up_52(conval_26)
        route_52 = torch.cat((up_52,RB_52),dim=1)
        conval_52 = self.contact_52(route_52)
        out_52= self.out_52(conval_52)

        up_104 = self.up_104(conval_52)
        route_104 = torch.cat((up_104,RB_104),dim=1)
        conval_104 = self.contact_104(route_104)
        out_104= self.out_104(conval_104)


        
        return  out_13,out_26,out_52,out_104



class Conv2d_Block_5L(nn.Module):#6个conv+bn+leakyReLU
    def __init__(self,in_channels,out_channels):
        super(Conv2d_Block_5L, self).__init__()
        self.Conv=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
            ConvolutionalLayer(in_channels,out_channels,1,1,0)#卷积5次
        )
    def forward(self,x):
        return self.Conv(x)


# 搭建输出网络
# class yolov3(nn.Module):
#     def __init__(self):
#         super(yolov3, self).__int__()
#         self.out_1=nn.Sequential(
#             DarkNet53[0],
#             Conv2d_Block_5L(52,512),#输入52，输出512
#             ConvolutionalLayer(512,52,3,1,1),
#             ConvolutionalLayer(52,33,3,1,1)#33为（5+种类数）*3
#         )
#         self.out_2=nn.Sequential(
#             #route_26 = torch.cat((up_26,h_26),dim=1)
#             torch.cat((DarkNet53(1),DarkNet53(3)),dim=1),
#             Conv2d_Block_5L(512,256),#输入512，输出256
#             ConvolutionalLayer(256,512,3,1,1),
#             ConvolutionalLayer(512,33,3,1,1)#33为（5+种类数）*3
#         )
#         self.out_3=nn.Sequential(
#             #route_26 = torch.cat((up_26,h_26),dim=1)
#             torch.cat((DarkNet53(2),DarkNet53(4)),dim=1),
#             Conv2d_Block_5L(256,128),#输入512，输出256
#             ConvolutionalLayer(128,256,3,1,1),
#             ConvolutionalLayer(256,33,3,1,1)#33为（5+种类数）*3
#         )
#         def forward(self,x):
#             out_1=self.out_1(x)
#             out_2=self.out_2(x)
#             out_3=self.out_3(x)

#             return out_1,out_2,out_3

# X = np.empty
# X=yolov3()


#结构可视化
# input=torch.rand(32,3,28,28)
# model=DarkNet53()

# with SummaryWriter(log_dir='logs',comment='data\DarkNet53') as w:
#     w.add_graph(model,(input,))


#输出尺寸
# model = DarkNet53()

# tfms = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open(r'E:\document\syn\data\graduation-project\yolo3-pytorch-master\VOCdevkit\VOC2007\img_18242.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 416, 416])

# out_7,out_14,out_28,out_56 = model(img)
# print(out_7.shape)
# print(out_14.shape)
# print(out_28.shape)
# print(out_56.shape)
