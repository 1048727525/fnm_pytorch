#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   WGAN_GP.py
@Time    :   2020/10/18 20:00:15
@Author  :   Wang Zhuo 
@Contact :   1048727525@qq.com
'''
import torch.nn as nn

#for demo
from util import *
import cv2

class Decoder(nn.Module):
    """Decoder part of generator

    """
    def __init__(self, input_dim=512, repeat_num=4, use_bias=False):
        super(Decoder, self).__init__()
        layers = []
        curr_dim = input_dim
        for i in range(repeat_num):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(ResnetBlock(curr_dim//2, use_bias=False))
            curr_dim = curr_dim // 2

        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=1, stride=1, padding=0, bias=use_bias))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
        

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.netD_eye = PixelDiscriminator(input_dim)
        self.netD_nose = PixelDiscriminator(input_dim)
        self.netD_mouth = PixelDiscriminator(input_dim)
        self.netD_face = PixelDiscriminator(input_dim)
        self.netD_map = PixelDiscriminator(input_dim)

        self.liner_eye = nn.Linear(1792, 1)
        self.liner_nose = nn.Linear(1536, 1)
        self.liner_mouth = nn.Linear(768, 1)
        self.liner_face = nn.Linear(16384, 1)
        self.liner_map = nn.Linear(43264, 1)

    def forward(self, x):
        ###################### Note ########################
        # Four Fixed Area. Modify them to fit your dataset #
        ####################################################
        '''
        eye = x[:, :, 64:100, 50:174].clone()
        nose = x[:, :, 75:140, 90:134].clone()
        mouth = x[:, :, 140:170, 75:149].clone()
        face = x[:, :, 64:180, 50:174].clone()
        '''

        eye = x[:, :, 56:102, 44:180].clone()
        nose = x[:, :, 70:144, 88:136].clone()
        mouth = x[:, :, 144:180, 80:144].clone()
        face = x[:, :, 40:190, 34:190].clone()
        whole_map = x.clone()
        '''
        cv2.imwrite("./demo_img/eye.jpg", cv2.cvtColor(tensor2im(eye), cv2.COLOR_BGR2RGB))
        cv2.imwrite("./demo_img/nose.jpg", cv2.cvtColor(tensor2im(nose), cv2.COLOR_BGR2RGB))
        cv2.imwrite("./demo_img/mouth.jpg", cv2.cvtColor(tensor2im(mouth), cv2.COLOR_BGR2RGB))
        cv2.imwrite("./demo_img/face.jpg", cv2.cvtColor(tensor2im(face), cv2.COLOR_BGR2RGB))
        '''
        D_eye = self.netD_eye(eye)
        D_nose = self.netD_nose(nose)
        D_mouth = self.netD_mouth(mouth)
        D_face = self.netD_face(face)
        D_map = self.netD_map(whole_map)

        D_eye = self.liner_eye(D_eye.view(D_eye.shape[0], -1))
        D_nose = self.liner_nose(D_nose.view(D_nose.shape[0], -1))
        D_mouth = self.liner_mouth(D_mouth.view(D_mouth.shape[0], -1))
        D_face = self.liner_face(D_face.view(D_face.shape[0], -1))
        D_map = self.liner_map(D_map.view(D_map.shape[0], -1))
        return D_face, D_eye, D_nose, D_mouth, D_map

    def get_partial_map(self, x):
        eye = x[:, :, 56:102, 44:180].clone()
        nose = x[:, :, 70:144, 88:136].clone()
        mouth = x[:, :, 144:180, 80:144].clone()
        face = x[:, :, 40:190, 34:190].clone()
        whole_map = x.clone()
        return face, eye, nose, mouth, whole_map


class PixelDiscriminator(nn.Module):
    def __init__(self, input_dim, ndf=32, depth=4):
        super(PixelDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_dim, ndf, kernel_size=3, stride=2, padding=0, bias=True))
        layers.append(nn.InstanceNorm2d(ndf))
        layers.append(nn.LeakyReLU(0.2, True))

        curr_dim = ndf
        for i in range(depth-1):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=0, bias=True))
            layers.append(nn.InstanceNorm2d(curr_dim))
            layers.append(nn.LeakyReLU(0.2, True))
            curr_dim = curr_dim*2
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

    