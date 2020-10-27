# -*- encoding: utf-8 -*-
# @File    :   se50_net.py
# @Time    :   2020/10/03 16:53:03
# @Author  :   Wang Zhuo 
# @Version :   1.0


import torch
import torch.nn as nn
from senet_model import Backbone


class se50_net(nn.Module):
    def __init__(self, model_path):
        super(se50_net, self).__init__()
        self.model = Backbone(50, 0.5, "ir_se")
        for p in self.model.parameters():
            p.requires_grad = False
        pre = torch.load(model_path,  map_location="cpu")
        self.model.load_state_dict(pre)
        self.model.eval()


    def get_feature(self, x):
        feature=self.model(x)
        norm = torch.norm(feature, 2, (1, 2, 3), True)
        feature = torch.div(feature, norm)
        '''
        print(feature.shape)
        norm = torch.norm(feature, 2, 1, True)
        print(norm)
        feature = torch.div(feature, norm)
        '''
        return feature
        
    def get_layers(self, x, num):
        return self.model.get_layers(x, num)

    def get_feature_vec(self, x):
        feature=self.model.get_fea(x)
        return feature



if __name__ == '__main__':
    def im2tensor(img):
        test_transform = transforms.Compose([
            transforms.Resize((img.shape[0], img.shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        img = transforms.ToPILImage()(img)
        return test_transform(img).unsqueeze(0).cuda()

    import cv2
    from PIL import Image
    from torchvision import transforms
    img_path = "../../dataset/dark_and_norm/testA/326_509.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = im2tensor(img)
    model = se50_net("model_ir_se50.pth").cuda()
    print(model.get_feature(img_tensor))

