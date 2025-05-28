import pandas as pd
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch import nn
from util.data_utils import CDDataAugmentation
import torchvision.models as models
import torch.nn.functional as F
#from nets.basicUnet_new import UNetTaskAligWeight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
from util.roi import process_and_augment_roi
from nets.basicUnet_new import UNetTaskAligWeight
import pywt

def wavelet_enhance(gray_img, wavelet='haar', level=1):
    """
    将灰度图像转换为小波增强后的伪RGB图像。

    参数:
        gray_img: numpy array，单通道灰度图像 (H, W)
        wavelet: 小波类型，默认为 'haar'
        level: 小波分解层数，默认为 1

    返回:
        img_rgb: numpy array，形状为 (3, H, W)，伪RGB图像
    """
    if len(gray_img.shape) == 3:
        gray_img = gray_img[0]  # 如果是 (1, H, W) 形式

    # 归一化到 [0, 255]，并转换为 float32
    gray_img = gray_img.astype(np.float32)
    if gray_img.max() <= 1.0:
        gray_img *= 255.0

    # 小波分解
    coeffs = pywt.wavedec2(gray_img, wavelet=wavelet, level=level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]

    # 重构高频图像（细节部分）
    high_freq = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)
    high_freq = cv2.resize(high_freq, gray_img.shape[::-1])  # 缩放回原图大小

    # 重构低频图像（主要结构）
    low_freq = cA
    low_freq = cv2.resize(low_freq, gray_img.shape[::-1])  # 同样缩放

    # 标准化各通道 [0, 255]
    def normalize(x):
        x = x - np.min(x)
        if np.max(x) != 0:
            x = x / np.max(x)
        return (x * 255).astype(np.uint8)

    R = normalize(gray_img)
    G = normalize(low_freq)
    B = normalize(high_freq)

    # 拼接为 RGB 格式，并转为 (3, H, W)
    img_rgb = np.stack([R, G, B], axis=0)

    return img_rgb
class GoogLeNetClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(GoogLeNetClassifier, self).__init__()
        # 加载预训练的 GoogLeNet
        self.googlenet = models.googlenet(pretrained=True)
        # 替换最后的全连接层
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)
def inference_all(model, test_loader, device, save_dir="test_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    records = []  # 用于保存所有预测结果

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            imgs = data['image'].float().to(device)
            filenames=data['filename']
            cl_out = model(imgs)
            # 分类处理
            predict_cl = torch.argmax(F.softmax(cl_out, dim=1), dim=1).cpu().numpy()

            for i in range(imgs.size(0)):
                name = filenames[i].replace('.png', '')  # 去除.png
                records.append(f"{name} {int(predict_cl[i])}")

    records.sort(key=lambda x: int(x.split()[0].replace('.jpg', '').replace('.png', '')))
    # 保存为 txt 文件
    with open(os.path.join(save_dir, "result.txt"), 'w') as f:
        for line in records:
            f.write(line + '\n')


import os
from PIL import Image
from torch.utils.data import Dataset


class TestImageDataset(Dataset):
    def __init__(self, model,image_dir):
        # 存储图片目录和变换方法
        self.image_dir = image_dir
        self.model=model
        self.image_names = sorted(os.listdir(image_dir))  # 按字母排序图片文件名
        self.augm1=CDDataAugmentation(img_size=224,
                                           ori_size=224, crop=None,p_hflip=0.0,p_vflip=0.0, color_jitter_params=None,
                                           long_mask=True)
        self.augm = CDDataAugmentation(img_size=224,
                                           ori_size=224, crop=None, p_hflip=0.0, p_vflip=0.0,
                                           color_jitter_params=None,
                                           long_mask=True)

    def __len__(self):
        # 返回图片的总数量
        return len(self.image_names)

    def __getitem__(self, idx):
        # 获取图片的文件名和路径
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # 打开图片并转换为RGB格式
        image = cv2.imread(img_path, 0)
        image = wavelet_enhance(image)
        image = np.transpose(image, (1, 2, 0))
        image=self.augm1.transform(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image,_ = process_and_augment_roi(self.model,image,device,self.augm,img_name)
        # 返回图片和文件名的字典
        return {'image': image, 'filename': img_name}



# 数据加载
model1 = UNetTaskAligWeight(n_channels=3, n_classes=1).to(device)
checkpoint = torch.load("checkpoint/Zhou1/best_model_epoch173.pt", map_location=device)
model1.load_state_dict(checkpoint['net'])
model1.to(device)
test_dataset = TestImageDataset(model1,image_dir='BUSI/test/TestSetA')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
# 加载模型
# model = UNetTaskAligWeight(n_channels=1,n_classes=1).to(device)
model = GoogLeNetClassifier(num_classes=6)
model = model.to(device)
# checkpoint= torch.load("trained_model/Zhou4/model_epoch120.pt", map_location=device)
checkpoint = torch.load("checkpoint/Zhou/best_acc_model_epoch1.pt", map_location=device)
model.load_state_dict(checkpoint['net'])
model.to(device)

if __name__ == '__main__':

    inference_all(model, test_loader, device, save_dir="test_results")





