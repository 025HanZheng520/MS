import os.path # gpu的路径
from numpy.random import randint #生成随机函数
from torch.utils import data #dataset类 __init__初始化；__getitim__方法获取txt文件
import glob # 使用文件路径的方法
import os
from dataloader.video_transform import *  #dataloader提供batch_size,shuffle,num_workers操作 transfrom发生在数据库geitim中
import numpy as np
import torchvision.transforms as transforms # datasets数据加载类，model常用预训练模型类，transforms图片转换，裁剪等
import random


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): #定义文件路径[0]
        return self._data[0]

    @property
    def num_frames(self): # 定义总帧数[1]
        return int(self._data[1])

    @property
    def label(self): # 定义表情标签[2]
        return int(self._data[2])


class VideoDataset(data.Dataset): # 数据集加载
    ## 数据的一些初始化操作和数据增强
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size):# 数据加载的一些参数
        self.list_file = list_file #数据加载路径、总帧数、标签
        self.duration = duration # 视频数据时长s,2
        self.num_segments = num_segments # 每个视频的采样的frames帧数:8
        self.transform = transform ##transform=test_transform,
        self.image_size = image_size #image大小定义112*112
        self.mode = mode #数据模式train or test
        self._parse_list() #定义函数，查看帧数是否>=16
        ##data augmentation
        self.brightness=transforms.Compose([ # compose串联多个图片变换操作

        transforms.ColorJitter(brightness=0.6), # 对图像颜色对比度、饱和度和零度进行变换
        transforms.RandomRotation(4), # 随机旋转

    ])

        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)] # txt按 划分，并删除' '，封装成数组
        # print(tmp)
        tmp = [item for item in tmp if int(item[1]) >= 16]  #tmp为>=16的num_frames
        self.video_list = [VideoRecord(item) for item in tmp] # item=row=_data=tmp
        self.soft_label =torch.zeros(len(self.video_list),7) # 按照frames>=16数目为row=item,形成7维标签的全是0的数组
        for i in range(len(self.video_list)):
            self.soft_label[i,self.video_list[i].label]=1
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record): #得到训练的指标

        self.num_segments=8
        self.duration=2
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets1 = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets1 = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets1 = np.zeros((self.num_segments,))

        return offsets1

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        self.num_segments = 8
        self.duration = 2
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
            print(segment_indices)

        return self.get(record, segment_indices,index)

    def get(self, record, indices,index):
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path.sort()

        images = list()
        # print(video_frames_path)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images =torch.reshape(images,(-1,3,self.image_size,self.image_size))
        """if self.mode == 'train':
            images = self.brightness(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        if record.label==2:
            if_neutral=1
        else:
            if_neutral=0"""


        # return images, record.label,index,if_neutral
        return images, record.label,index

    def __len__(self):
        return len(self.video_list)


def train_data_loader(data_set):
    image_size = 224
    train_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                           GroupRandomHorizontalFlip(),
                                                           Stack(),
                                                           ToTorchFormatTensor()])
    train_data = VideoDataset(list_file="./annotation/zh_set_" + str(data_set) + "_train.txt",  # 数据集加载
                                  num_segments=8,
                                  duration=2,
                                  mode='train',
                                  transform=train_transforms,
                                  image_size=image_size)
    return train_data


def test_data_loader(data_set):
    image_size = 224
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = VideoDataset(list_file="./annotation/zh_set_"+str(data_set)+"_test.txt",
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data

def create_dataloader(opt):
    train_data = train_data_loader(data_set=opt.data_set)
    test_data = test_data_loader(data_set=opt.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=opt.workers,
                                                  pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.workers,
                                                  pin_memory=True,
                                                  drop_last=True)



    return train_loader, val_loader
