import numpy as np
import random
import torch


# target one-hot编码
def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    shape = np.array(tensor.shape)
    shape[1] = n_classes
    shape = tuple(shape)
    one_hot = torch.zeros(shape).scatter_(1, tensor.cpu(), 1)
    return one_hot
def to_one_hot(tensor,n_classes):
    # 将原始标签列表转换为one-hot编码
    labels = torch.zeros(len(tensor), n_classes)
    labels[range(len(tensor)), tensor] = 1
    return labels
def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],z_random:z_random + crop_size[2]]

    return crop_img, crop_label

def center_crop_3d(img, label, slice_num=16):
    if img.shape[0] < slice_num:
        return None
    left_x = img.shape[0]//2 - slice_num//2
    right_x = img.shape[0]//2 + slice_num//2

    crop_img = img[left_x:right_x]
    crop_label = label[left_x:right_x]
    return crop_img, crop_label

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_V2(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr