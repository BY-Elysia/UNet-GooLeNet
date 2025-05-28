import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F


def process_and_augment_roi(model, image, device, transform_fn, name, padding=30):
    # image 是经过 resize 和 to_tensor 后的 tensor，shape = [3, H, W]
    _, h, w = image.shape
    # 模型推理，模型输入需要[batch, C, H, W]
    input_tensor = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        seg_out = model(input_tensor)
        se_out=seg_out
        seg_out = torch.sigmoid(seg_out)
        pred_mask = (seg_out > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

    ys, xs = np.where(pred_mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        # print("⚠️ 未检测到病灶区域，返回中心裁剪")
        center_x, center_y = w // 2, h // 2
        size = min(h, w) // 2
        x_min, x_max = center_x - size//2, center_x + size//2
        y_min, y_max = center_y - size//2, center_y + size//2
    else:
        x_min = max(xs.min() - padding, 0)
        x_max = min(xs.max() + padding, w)
        y_min = max(ys.min() - padding, 0)
        y_max = min(ys.max() + padding, h)

    # 裁剪 ROI，tensor 的裁剪
    roi_tensor = image[:, y_min:y_max, x_min:x_max]  # [3, H_roi, W_roi]

    # 转成 numpy rgb 图进行增强
    roi_np = roi_tensor.permute(1, 2, 0).cpu().numpy()  # [H_roi, W_roi, 3], float, 0~1
    roi_np = (roi_np * 255).astype(np.uint8)
    roi_rgb = cv2.cvtColor(roi_np, cv2.COLOR_BGR2RGB)
    # plt.imshow(roi_rgb)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    roi_tensor_aug = transform_fn.transform(roi_rgb)  # 增强后tensor

    return roi_tensor_aug,se_out


