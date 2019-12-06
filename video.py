# coding: utf-8
'''
载入显著图特征，用vgg16提取
'''

import os
import cv2
import h5py
import numpy as np
import skimage
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from IncepResv2 import InceptionResNetV2
from args import frame_shape
from args import video_root, video_sort_lambda
from args import feature_h5_path, feature_h5_feats, feature_h5_lens
from args import resnet_checkpoint, c3d_checkpoint, IRV2_checkpoint
from args import max_frames, feature_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AppearanceEncoder_resnet152(nn.Module):
    def __init__(self): # , embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(AppearanceEncoder_resnet152, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        return features


class AppearanceEncoder_inceptionresnetv2(nn.Module):
    def __init__(self):
        super(AppearanceEncoder_inceptionresnetv2, self).__init__()
        IRV2 = InceptionResNetV2(num_classes=1001)
        # print('IRV2:\n', IRV2)
        IRV2.load_state_dict(torch.load(IRV2_checkpoint))
        modules = list(IRV2.children())[:-1]  # delete the last fc layer.
        self.IRV2 = nn.Sequential(*modules)
        # print('IRV2:\n', self.IRV2)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.IRV2(images)
        features = features.reshape(features.size(0), -1)
        # print(features.size())
        return features

class AppearanceEncoder_resnet50(nn.Module):

    # 使用ResNet50作为视觉特征提取器
    def __init__(self):
        super(AppearanceEncoder_resnet50, self).__init__()
        self.resnet = models.resnet50()
        self.resnet.load_state_dict(torch.load(resnet_checkpoint))
        del self.resnet.fc

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class AppearanceEncoder_inception3(nn.Module):
    def __init__(self): # , embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(AppearanceEncoder_inception3, self).__init__()
        inception3 = models.inception_v3(pretrained=True)
        modules = list(inception3.children())[:-1]  # delete the last fc layer.
        self.inception3 = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.inception3(images)
        features = features.reshape(features.size(0), -1)
        return features

class C3D(nn.Module):
    '''
    C3D model (https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py)
    '''

    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        return x


class MotionEncoder(nn.Module):

    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.c3d = C3D()
        pretrained_dict = torch.load(c3d_checkpoint)
        model_dict = self.c3d.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.c3d.load_state_dict(model_dict)

    def forward(self, x):
        return self.c3d(x)


def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # 把BGR的图片转换成RGB的图片，因为之后的模型用的是RGB格式
        if ret is False:
            break
        frame = frame[:, :, ::-1] #::-1翻转
        frames.append(frame)
        frame_count += 1

    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height), target_height))
        # print(resized_image.shape)
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        # print(cropping_length)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
        # print(resized_image.shape)
    else:
        resized_image = cv2.resize(image, (target_width, int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length: resized_image.shape[0] - cropping_length]

    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    # image -= np.array([0.485, 0.456, 0.406])
    # image /= np.array([0.229, 0.224, 0.225])
    image -= np.array([0.5, 0.5, 0.5])
    image /= np.array([0.5, 0.5, 0.5])
    return image


def extract_features(aencoder, mencoder,device):
    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=video_sort_lambda)
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    feature_size = 1536
    if os.path.exists(feature_h5_path):
        # 如果hdf5文件已经存在，说明之前处理过，或许是没有完全处理完
        # 使用r+ (read and write)模式读取，以免覆盖掉之前保存好的数据
        h5 = h5py.File(feature_h5_path, 'r+')
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, 'w')
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # 提取视频帧以及视频小块
        frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        frame_list = np.array([preprocess_frame(x, frame_shape[1], frame_shape[2]) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        # 先提取表观特征
        with torch.no_grad():
            frame_list = Variable(torch.from_numpy(frame_list)).to(device)
            # print(frame_list.size())
            af = aencoder(frame_list)

        # 再提取动作特征
        # clip_list = np.array([[resize_frame(x, 112, 112)
        #                        for x in clip] for clip in clip_list])
        # clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
        # with torch.no_grad():
        #     clip_list = Variable(torch.from_numpy(clip_list)).to(device)
        #     mf = mencoder(clip_list)
        #
        # # 视频特征的shape是max_frames x (2048 + 4096)
        # # 如果帧的数量小于max_frames，则剩余的部分用0补足
        # feats = np.zeros((max_frames, feature_size), dtype='float32')
        # # feats = np.zeros((max_frames, 6144), dtype='float32')
        #
        # # 合并表观和动作特征
        # # print(af.size(),mf.size())
        feats = af.data.cpu().numpy()
        # feats[:frame_count, :] = torch.cat([af, mf], dim=1).data.cpu().numpy()
        # feats[:frame_count, :] = af.data.cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count


def main():
    print('device: ',DEVICE)
    # print('Extracting video feature by resnet152')
    # aencoder = AppearanceEncoder_resnet152()
    print('Extracting video feature by inceptionresnetv2')
    aencoder = AppearanceEncoder_inceptionresnetv2()
    # aencoder = EncoderCNN()
    aencoder.eval()
    aencoder.to(DEVICE)

    mencoder = MotionEncoder()
    mencoder.eval()
    mencoder.to(DEVICE)

    extract_features(aencoder, mencoder,DEVICE)


if __name__ == '__main__':
    main()
