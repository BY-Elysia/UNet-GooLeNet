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
from nets.basicUnet_new import UNetTaskAligWeight
import torchvision.models as models
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
from util.roi import process_and_augment_roi
from util.loss import DC_and_BCE_loss
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pywt
import cv2


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
class CustomDataset(Dataset):
    def __init__(self,model, list,path, img_size=512,crop=None,is_train=None,):
        self.image_list = list
        self.image_folder = os.path.join(path,'images')
        self.label_folder = os.path.join(path,'labels','label.txt')
        self.img_size = img_size
        self.model=model
        self.augm1=CDDataAugmentation(img_size=224,
                                           ori_size=224, crop=crop,p_hflip=0.0,p_vflip=0.0, color_jitter_params=None,
                                           long_mask=True)

        # 读取图像名和标签
        self.image_list = []
        self.cl_labels = []
        with open(self.label_folder, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.image_list.append(img_name)
                self.cl_labels.append(int(label))

        if is_train:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=True, p_hflip=0.6,p_vflip=0.5, p_rota=0.6, p_scale=0.6,
                                           p_gaussn=0.6,
                                           p_contr=0.6, p_gama=0.6, p_distor=0.6, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                                           p_random_affine=0,
                                           long_mask=True)  # image reprocessing

        else:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=crop,p_hflip=0.0,p_vflip=0.0, color_jitter_params=None,
                                           long_mask=True)
    def __len__(self):
        return len(self.image_list)
    # def correct_dims(self,*images):
    #     corr_images = []
    #     # print(images)
    #     for img in images:
    #         if len(img.shape) == 2:
    #             corr_images.append(np.expand_dims(img, axis=2))
    #         else:
    #             corr_images.append(img)
    #     if len(corr_images) == 1:
    #         return corr_images[0]
    #     else:
    #         return corr_images
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_list[idx])
        image = cv2.imread(img_name, 0)
        image = wavelet_enhance(image)
        image = np.transpose(image, (1, 2, 0))
        # image = np.asarray(Image.open(img_name).convert('RGB'))
        # se_name = os.path.join(self.label_folder, self.image_list[idx])
        # se_label = cv2.imread(se_name, 0)
        # se_label = np.asarray(Image.open(se_name),dtype=np.uint8)
        # se_label = se_label // 255
        cl_label = self.cl_labels[idx]
        # image, se_label = self.correct_dims(image, se_label)
        image=self.augm1.transform(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image,se_out = process_and_augment_roi(self.model,image,device,self.augm,img_name)
        se_out = se_out.squeeze(0)
        return {
            "image": image,
            'cl_label':cl_label,
            'se_out':se_out
        }


def val(model,val_loader, loss_cl, device):
    model.eval()
    running_loss = 0.0
    cl_total_loss = 0.0
    f1_metric = F1Score(num_classes=6, average='macro', task='multiclass').to(device)
    acc_metric = Accuracy(num_classes=6, average='macro', task='multiclass').to(device)
    auc_metric = AUROC(num_classes=6, average='macro', task='multiclass').to(device)
    matrix2 = ConfusionMatrix(num_classes=6, task='multiclass').to(device)
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            imgs = data['image'].float().to(device)
            cl_label = data['cl_label'].long().to(device)
            cl_out = model(imgs)
            cl_out = torch.squeeze(cl_out,dim=1)
            criterion = torch.nn.CrossEntropyLoss()
            cl_loss = criterion(cl_out, cl_label)
            cl_total_loss += cl_loss
            loss =  cl_loss
            running_loss += loss.item()
            predict_cl = torch.argmax(F.softmax(cl_out, dim=-1), dim=-1).to(device)
            auc_metric(cl_out, cl_label)
            f1_metric(predict_cl, cl_label)
            acc_metric(predict_cl, cl_label)
            matrix2(predict_cl, cl_label)
    print("cl - {:.4f}".format(  cl_total_loss / len(val_loader)))
    f1_score = f1_metric.compute().cpu().numpy()
    acc_score = acc_metric.compute().cpu().numpy()
    auc_score = auc_metric.compute().cpu().numpy()
    matrix2_score = matrix2.compute().cpu().numpy()
    torch.cuda.empty_cache()
    return running_loss / len(val_loader),f1_score,acc_score,auc_score,matrix2_score


def train(model,optimizer, train_loader,loss_cl,device,epoch,n=2):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    running_loss = 0.0
    cl_total_loss = 0.0
    tempPred = 0
    for idx, data in enumerate(train_loader):
        imgs = data['image'].float().to(device)
        cl_label = data['cl_label'].long().to(device)
        se_out=data['se_out'].long().to(device)
        # cl_out = model(imgs)
        for i in range(n):
            if i == 0:
                cl_out = model(imgs)
                tempPred = se_out.detach()
            else:
                tempPred = F.sigmoid(tempPred).to(device)
                # diff = custom_function(tempPred,device)
                # pred_masks = torch.where(tempPred > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                # diff = se_label -pred_masks
                batch_size, num_channels, height, width = tempPred.size()
                # feature_map_flat = tempPred.view(batch_size, num_channels, -1)  # 平均池化，得到 (batch_size, num_channels)
                # weight = model.weight(feature_map_flat).view(batch_size, 1, 1,1)  # 计算权重，并调整为 (batch_size, 1, 1, 1)
                # projected_images = imgs+tempPred*weight
                diff = (torch.abs(0.5 - tempPred) * 2).view(batch_size, -1)
                diff = torch.sum(diff, dim=1) / (diff.size()[1])
                diff = diff.view(batch_size, 1, 1, 1)
                imgs = imgs + tempPred * diff
                cl_out = model(imgs)
            cl_out = torch.squeeze(cl_out, dim=1)
            criterion = torch.nn.CrossEntropyLoss()
            cl_loss = criterion(cl_out, cl_label)
            #
            loss = cl_loss
            # loss = model.loss_function.cuda()(se_loss, cl_loss)
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            cl_total_loss += cl_loss
            running_loss += loss.item()
    print(" cl - {:.4f}".format( cl_total_loss / len(train_loader)/n))
    torch.cuda.empty_cache()
    return running_loss / len(train_loader)/n



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
    train_folder = 'BUSI/train'
    val_folder = 'BUSI/val'
    train_image_list = sorted(os.listdir(os.path.join(train_folder, 'images')))
    val_image_list = sorted(os.listdir(os.path.join(val_folder, 'images')))
    device = torch.device('cpu' if args.cpu else 'cuda')

    import shutil
    save_path = os.path.join("checkpoint", "Zhou")

    # 删除目录及其内容
    if os.path.exists(save_path):
        try:
            shutil.rmtree(save_path)  # 删除目录及其中的所有内容
        except Exception as e:
            print(f"删除目录失败: {e}")
            # 这里可以进行其他错误处理

    # 重新创建目录
    os.makedirs(save_path)
    # writer = SummaryWriter(log_dir=save_path)

    lr_threshold = 0.0001
    seed_everything(args.seed)
    model1 = UNetTaskAligWeight(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load("checkpoint/Zhou1/best_model_epoch173.pt", map_location=device)
    model1.load_state_dict(checkpoint['net'])
    model1.to(device)
    train_dataset = CustomDataset(model1,train_image_list, train_folder, img_size=args.img_size, is_train=True)
    val_dataset = CustomDataset(model1,val_image_list, val_folder, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # model = UNetTaskAligWeight(n_channels=1,n_classes=1).to(device)
    model = GoogLeNetClassifier(num_classes=6)
    model = model.to(device)
    checkpoint = torch.load("checkpoint/Zhou6/0.8605/best_model_epoch1.pt", map_location=device)
    model.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001,
                                  threshold_mode="abs", min_lr=0.00001)  # gamma是衰减因子
    cl_loss = loss.BCEFocalLoss(gamma=2,alpha=0.4).to(device)

    start_epoch = 0

    best_val_index = 0
    best_min_loss = 2
    best_train_loss = 2
    early_stop_patience = 300  # 设置早停的阈值
    early_stop_counter = 0
    best_model_path = ''
    best_seg_model_path =' '
    best_train_model_path=' '
    model_path=''
    for epoch in range(start_epoch,args.epochs):

        train_loss = train(model,optimizer, train_loader,cl_loss,device, epoch)
        val_loss, f1_score,acc_score,auc_score,matrix2_score = val(model,val_loader,cl_loss,device)

        scheduler.step(train_loss)

        print(f"Epoch_[{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(
              f" f1: {f1_score:.4f}, acc: {acc_score:.4f} ,auc: {auc_score:.4f},,matrix_score2:{matrix2_score}",flush=True)
        index =  acc_score


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
            best_seg_model_path = f"{save_path}/best_acc_model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_seg_model_path)
            os.chmod(best_seg_model_path, 0o777)

        if epoch %10==0:
            # try:
            #     os.remove()
            # except OSError as e:
            #     print(OSError)
            model_path = f"{save_path}/model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path)
            os.chmod(model_path, 0o777)
    del model
    # 关闭TensorBoard记录器
    # writer.close()

if __name__ == '__main__':
    args = config.args

    # 进行交叉验证
    cross_validation(args)
