import pandas as pd
import torch
import cv2
import numpy as np
from tqdm import tqdm
from util.data_utils import CDDataAugmentation
import torch.nn.functional as F
from nets.basicUnet import UNetTaskAligWeight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
def inference_all(model, test_loader, device, save_dir="test_results"):
    model.eval()
    seg_dir = os.path.join(save_dir, "Segmentation_Results")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    records = []  # 用于保存所有预测结果

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            imgs = data['image'].float().to(device)
            filenames=data['filename']
            se_out = model(imgs)

            # 分割处理
            se_out = torch.sigmoid(se_out)
            pred_masks = (se_out > 0.5).float()

            # # 分类处理
            # predict_cl = torch.argmax(F.softmax(cl_out, dim=1), dim=1).cpu().numpy()

            for i in range(imgs.size(0)):
                name = filenames[i].replace('.jpg', '')  # 去除.png
                mask = pred_masks[i].squeeze().cpu().numpy().astype('uint8')

                h, w = mask.shape
                rgb_image = Image.new("RGB", (w, h), (0, 0, 0))

                red = (255, 0, 0)
                for y in range(h):
                    for x in range(w):
                        if mask[y, x] == 1:
                            rgb_image.putpixel((x, y), red)
                # rgb_image=rgb_image.resize((512,512))
                rgb_image.save(os.path.join(seg_dir, f"{name}.png"))
                # # 添加到结果记录
                # records.append({'Image': name, 'Pterygium': int(predict_cl[i])})

    # 保存为 Excel
    df = pd.DataFrame(records)
    df.to_excel(os.path.join(save_dir, "Classification_Results.xlsx"), index=False)


import os
from PIL import Image
from torch.utils.data import Dataset


class TestImageDataset(Dataset):
    def __init__(self, image_dir):
        # 存储图片目录和变换方法
        self.image_dir = image_dir
        self.image_names = sorted(os.listdir(image_dir))  # 按字母排序图片文件名
        self.augm = CDDataAugmentation(img_size=224,
                                           ori_size=224, crop=None, p_hflip=0.0, p_vflip=0.0,
                                           color_jitter_params=None,
                                           long_mask=True)
    def correct_dims(self, *images):
        corr_images = []
        for img in images:
            if len(img.shape) == 2:
                corr_images.append(np.expand_dims(img, axis=2))
            else:
                corr_images.append(img)
        if len(corr_images) == 1:
            return corr_images[0]
        else:
            return corr_images

    def __len__(self):
        # 返回图片的总数量
        return len(self.image_names)

    def __getitem__(self, idx):
        # 获取图片的文件名和路径
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # 打开图片并转换为RGB格式
        image = cv2.imread(img_path, 1)
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[1:] if image.ndim == 3 else image.shape
            mask = torch.zeros((height, width), dtype=torch.uint8)
        else:
            raise TypeError("Unsupported image type")
        # 如果有指定变换，则应用变换
        image= self.correct_dims(image)
        image,_ = self.augm.transform(image,mask)

        # 返回图片和文件名的字典
        return {'image': image, 'filename': img_name}


# 数据加载
test_dataset = TestImageDataset(image_dir='BUSI_1/newval')

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# 加载模型
model = UNetTaskAligWeight(n_channels=3,n_classes=1).to(device)
checkpoint = torch.load("checkpoint/Zhou/best_model_epoch155.pt", map_location=device)
model.load_state_dict(checkpoint['net'])
model.to(device)

if __name__ == '__main__':
    inference_all(model, test_loader, device, save_dir="test_results")





