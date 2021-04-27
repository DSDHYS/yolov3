import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolov3_tiny import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.config import Config
from utils.dataloader import YoloDataset, yolo_dataset_collate

from tensorboardX import SummaryWriter


def get_lr(optimizer):#优化器，设置每个参数组的学习率
    for param_group in optimizer.param_groups:#optimizer.param_groups： 是长度为2的list，其中的元素是2个字典
        return param_group['lr']
        
def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0

    net.train()#启用训练模式
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:#进度条
        for iteration, batch in enumerate(gen):#enunmerate为可遍历对象创建索引序列
            if iteration >= epoch_size:#iteration迭代器
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()#数组读入图片，并转换为张量
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]


            optimizer.zero_grad()#清零梯度
            outputs = net(images)#前向传播
            losses = []
            num_pos_all = 0

            for i in range(3):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos#计算损失

            loss = sum(losses) / num_pos

            loss.backward()#反向传播
            optimizer.step()
            
            #损失可视化
            writer.add_scalar('训练损失值',loss,epoch)

            total_loss += loss.item()


            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), #后缀样式
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))

    #torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    log.append(total_loss/(epoch_size+1))
    #print(log)
    if epoch%20==0:
        torch.save(model.state_dict(), 'logs/Epoch%d.pth'%(epoch))#保存
    else:
        torch.save(model.state_dict(), 'logs/Epoch.pth')#保存

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    log=[]

    Cuda = True

    writer=SummaryWriter(log_dir='view',comment='Linear')

    Use_Data_Loader = True#图片预处理

    normalize = True#损失归一化处理

    model = YoloBody(Config)#引用神经网络模型

    # model_path = "model_data/train.pth"#预训练权重
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#GPU或CPU选择
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    net = model.train()
    #print(net)

    if Cuda:
        net = torch.nn.DataParallel(model)#数据并行处理
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]),#nn.shape将输入转化为张量,-1代表自动计算
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda, normalize))
        


    annotation_path = 'train.txt'#图片所在路径
  
    val_split = 0.1#验证集占比
    with open(annotation_path,encoding='utf-8') as f:
        lines = f.readlines()#行数
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)#计算验证集数
    num_train = len(lines) - num_val#计算训练集数
    
    if True:
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch =0
        Unfreeze_Epoch = 400

        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)#基于周期数调整学习率
        
        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]), True)
            val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]), False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)#组合数据集和采样器
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                             (Config["img_h"], Config["img_w"])).generate(True)
            gen_val = Generator(Batch_size, lines[num_train:],
                             (Config["img_h"], Config["img_w"])).generate(False)
                        
        epoch_size = num_train//Batch_size #训练次数
        epoch_size_val = num_val//Batch_size

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):#训练周期
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)#训练
            lr_scheduler.step()
        log_last=np.array(log)
        log_last=log_last.reshape(-1,1)
        print(log_last)
            
            
