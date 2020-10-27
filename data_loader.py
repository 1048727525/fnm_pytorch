import os
import scipy
import numpy as np
from util import *
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class sample_dataset(Dataset):
    def __init__(self, list_path, img_root_path, crop_size, image_size, mode="train"):
        self.img_name_list = read_txt_file(list_path)
        self.img_root_path = img_root_path
        transform = []
        if mode == "train":
            transform.append(transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        transform_112 = []
        if mode == "train":
            transform_112.append(transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
            transform_112.append(transforms.RandomHorizontalFlip())
        transform_112.append(transforms.CenterCrop(crop_size))
        transform_112.append(transforms.Resize(112))
        transform_112.append(transforms.ToTensor())
        transform_112.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform_112 = transforms.Compose(transform_112)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root_path, self.img_name_list[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), self.transform_112(img)

def get_loader(list_path, img_root_path, crop_size=224, image_size=224, batch_size=16, mode="train", num_workers=8):
    dataset = sample_dataset(list_path, img_root_path, crop_size, image_size)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return data_loader

        
if __name__ == '__main__':
    import cv2
    
    profile_list_path = "../fnm/mpie/casia_gt.txt"
    front_list_path = "../fnm/mpie/session01_front_demo.txt"
    profile_path = "../../datasets/casia_aligned_250_250_jpg"
    front_path = "../../datasets/session01_align"
    crop_size = 224
    image_size = 224
    #dataset = sample_dataset(profile_list_path, profile_path, crop_size, image_size)
    '''
    for i, sample in enumerate(dataset):
        cv2.imwrite("profile.jpg", tensor2im(sample["profile"]))
        cv2.imwrite("front.jpg", tensor2im(sample["front"]))
        if i==1:
            break
    '''
    data_loader = get_loader(front_list_path, front_path, crop_size=224, image_size=224, batch_size=16, mode="train", num_workers=8)
    for i, sample in data_loader:
        print(sample.shape)
    '''
    for i, sample in enumerate(data_loader):
        cv2.imwrite("profile.jpg", cv2.cvtColor(tensor2im(sample["profile"]), cv2.COLOR_BGR2RGB))
        cv2.imwrite("front.jpg", cv2.cvtColor(tensor2im(sample["front"]), cv2.COLOR_BGR2RGB))
        if i==1:
            break
    '''

