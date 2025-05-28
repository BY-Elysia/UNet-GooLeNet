import os
import warnings

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from monai.optimizers import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
from monai.utils import MetricReduction



import torch


import config


from torch import nn

from nets.basicUnet import UNetTaskAligWeight
from util import logger, metrics, common,loss
from torch.utils.data import DataLoader,random_split

import torch.nn.functional as F

from monai.metrics import DiceMetric, HausdorffDistanceMetric,MeanIoU
from torchmetrics import Recall, Precision, ConfusionMatrix, F1Score, Accuracy, AUROC

from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter



from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau

from random import  uniform,random



from monai.utils import set_determinism

import random

from util.data_utils import CDDataAugmentation
from util.loss import DC_and_BCE_loss

import warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    def __init__(self, list,path, img_size=224,crop=None,is_train=None):
        self.image_list = list
        self.image_folder = os.path.join(path,'images')
        self.label_folder = os.path.join(path,'labels')
        self.img_size = img_size
        if is_train:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=True, p_hflip=0.5,p_vflip=0.5, p_rota=0.5, p_scale=0.6,
                                           p_gaussn=0.5,
                                           p_contr=0.0, p_gama=0.5, p_distor=0.0, color_jitter_params=None,
                                           p_random_affine=0,
                                           long_mask=True)  # image reprocessing

        else:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=crop,p_hflip=0.0,p_vflip=0.0, color_jitter_params=None,
                                           long_mask=True)
    def __len__(self):
        return len(self.image_list)
    def correct_dims(self,*images):
        corr_images = []
        # print(images)
        for img in images:
            if len(img.shape) == 2:
                corr_images.append(np.expand_dims(img, axis=2))
            else:
                corr_images.append(img)
        if len(corr_images) == 1:
            return corr_images[0]
        else:
            return corr_images
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_list[idx])
        image = cv2.imread(img_name, 1)
        # image = np.asarray(Image.open(img_name).convert('RGB'))
        se_name = os.path.join(self.label_folder, self.image_list[idx])
        se_label = cv2.imread(se_name, 0)
        # se_label = np.asarray(Image.open(se_name),dtype=np.uint8)
        se_label = se_label // 255
        cl_label = int(self.image_list[idx][0])-1

        image, se_label = self.correct_dims(image, se_label)
        image, se_label = self.augm.transform(image, se_label)
        se_label = se_label.unsqueeze(0)

        return {
            "image": image,
            'cl_label':cl_label,
            'se_label':se_label,
        }


def val(model,val_loader, loss_se,loss_cl, device):
    model.eval()
    running_loss = 0.0
    seg_total_loss = 0.0
    dice_metric = DiceMetric(include_background=False,reduction=MetricReduction.MEAN)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",reduction=MetricReduction.MEAN)
    iou_metric = MeanIoU(include_background=False)
    with torch.no_grad():

        for idx, data in enumerate(val_loader):

            imgs = data['image'].float().to(device)

            se_label = data['se_label'].float().to(device)
            se_out= model(imgs)

            se_loss = loss_se(se_out, se_label).to(device)
            seg_total_loss += se_loss

            loss =  se_loss

            running_loss += loss.item()


            se_out = F.sigmoid(se_out)
            pred_masks = torch.where(se_out > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
            # 计算Dice指数

            for i in range(pred_masks.size(0)):
                if torch.all(pred_masks[i] == 0):
                    pred_masks[i][0][0][0] = 1
            dice_metric(pred_masks, se_label)
            # 计算Hausdorff距离
            hausdorff_metric(pred_masks, se_label)
            iou_metric(pred_masks, se_label)
    print("seg - {:.4f}".format( seg_total_loss / len(val_loader)))
    dice_score = dice_metric.aggregate().item()
    hausdorff_distance = hausdorff_metric.aggregate().item()
    iou_score = iou_metric.aggregate().item()
    torch.cuda.empty_cache()
    return running_loss / len(val_loader),dice_score,hausdorff_distance,iou_score,seg_total_loss/len(val_loader)


def train(model,optimizer, train_loader,loss_se,loss_cl,device,epoch,n=2):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    running_loss = 0.0
    seg_total_loss = 0.0
    tempPred = 0
    for idx, data in enumerate(train_loader):

        imgs = data['image'].float().to(device)
        se_label = data['se_label'].float().to(device)
        for i in range(n):
            if i == 0:
                se_out= model(imgs)
                tempPred = se_out.detach().cpu()
            else:
                tempPred = F.sigmoid(tempPred).to(device)
                # diff = custom_function(tempPred,device)
                # pred_masks = torch.where(tempPred > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                # diff = se_label -pred_masks
                batch_size, num_channels, height, width = tempPred.size()
                # feature_map_flat = tempPred.view(batch_size, num_channels, -1)  # 平均池化，得到 (batch_size, num_channels)
                # weight = model.weight(feature_map_flat).view(batch_size, 1, 1,1)  # 计算权重，并调整为 (batch_size, 1, 1, 1)
                # projected_images = imgs+tempPred*weight
                diff = (torch.abs(0.5-tempPred)*2).view(batch_size, -1)
                diff = torch.sum(diff,dim=1)/(diff.size()[1])
                diff = diff.view(batch_size, 1, 1,1)
                imgs=imgs +tempPred * diff
                se_out= model(imgs)
            se_loss = loss_se(se_out, se_label).to(device)
            loss = se_loss

            # loss = model.loss_function.cuda()(se_loss, cl_loss)
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            seg_total_loss += se_loss
            running_loss += loss.item()
    print(" seg - {:.4f}".format( seg_total_loss / len(train_loader)/n))
    torch.cuda.empty_cache()
    return running_loss / len(train_loader)/n,seg_total_loss / len(train_loader)




def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# 定义交叉验证函数
# 定义交叉验证函数
def cross_validation(args):
    train_folder = 'BUSI_1/train'
    val_folder = 'BUSI_1/val'
    train_image_list = sorted(os.listdir(os.path.join(train_folder, 'images')))
    val_image_list = sorted(os.listdir(os.path.join(val_folder, 'images')))

    device = torch.device('cpu' if args.cpu else 'cuda')
    import shutil
    save_path = os.path.join("checkpoint", "Zhou1")

    # 删除目录及其内容
    if os.path.exists(save_path):
        try:
            shutil.rmtree(save_path)  # 删除目录及其中的所有内容
        except Exception as e:
            print(f"删除目录失败: {e}")
            # 这里可以进行其他错误处理
    # 重新创建目录
    os.makedirs(save_path)
    lr_threshold = 0.0001
    seed_everything(args.seed)

    train_dataset = CustomDataset(train_image_list, train_folder, img_size=args.img_size, is_train=True)
    val_dataset = CustomDataset(val_image_list, val_folder, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNetTaskAligWeight(n_channels=3,n_classes=1).to(device)
    checkpoint = torch.load("checkpoint/Zhou/best_model_epoch155.pt", map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001,
                                  threshold_mode="abs", min_lr=0.00001)  # gamma是衰减因子

    se_loss = DC_and_BCE_loss(dice_weight=0.5).to(device)
    cl_loss = loss.BCEFocalLoss(gamma=2,alpha=0.4).to(device)

    start_epoch = 0

    best_val_index = 0
    best_min_loss = 2
    early_stop_patience = 50  # 设置早停的阈值
    early_stop_counter = 0
    best_model_path = ''
    best_seg_model_path =' '
    for epoch in range(start_epoch,args.epochs):

        train_loss,train_seg_loss = train(model,optimizer, train_loader,se_loss,cl_loss,device, epoch)
        val_loss, dice_score,hd_score,iou_score,val_seg_loss = val(model,val_loader,se_loss,cl_loss,device)

        scheduler.step(train_loss)

        print(f"Epoch_[{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch}/{args.epochs}] - Dice Score: {dice_score:.4f}  - Hd Score: {hd_score:.4f} - iou_score: {iou_score:.4f}"
             ,flush=True)
        index =  dice_score


        if val_loss < best_min_loss:
            best_min_loss = val_loss
            early_stop_counter = 0
            try:
                os.remove(best_model_path)
            except OSError as e:
                print(OSError)
            best_model_path = f"{save_path}/best_model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_model_path)
            os.chmod(best_model_path, 0o777)
        else:
            early_stop_counter += 1

        print(f"early_stop_counter_[{early_stop_counter}]")
        if early_stop_counter > early_stop_patience:
            if optimizer.param_groups[0]['lr'] >= lr_threshold:
                print("My patience ended, but I believe I need more time")
                early_stop_counter = early_stop_counter - 20
            else:
                print("Early stoping epoch!!", epoch)
                break
        if index > best_val_index:
            best_val_index = index
            try:
                os.remove(best_seg_model_path)
            except OSError as e:
                print(OSError)
            best_seg_model_path = f"{save_path}/best_seg_model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_seg_model_path)
            os.chmod(best_seg_model_path, 0o777)
    del model


if __name__ == '__main__':
    args = config.args

    # 进行交叉验证
    cross_validation(args)
