import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as F
from torchvision import transforms as T
import torch
from torchvision.transforms import InterpolationMode

def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:

    # def __init__(
    #         self,
    #         img_size,
    #         with_random_hflip=False,
    #         with_random_vflip=False,
    #         with_random_rot=False,
    #         with_random_crop=False,
    #         with_scale_random_crop=False,
    #         with_random_blur=False,
    # ):
    #     self.img_size = img_size
    #     if self.img_size is None:
    #         self.img_size_dynamic = True
    #     else:
    #         self.img_size_dynamic = False
    #     self.with_random_hflip = with_random_hflip
    #     self.with_random_vflip = with_random_vflip
    #     self.with_random_rot = with_random_rot
    #     self.with_random_crop = with_random_crop
    #     self.with_scale_random_crop = with_scale_random_crop
    #     self.with_random_blur = with_random_blur
    def __init__(self, img_size=512,  ori_size=512, crop=False, p_hflip=0.0, p_vflip=0.0,p_rota=0.0, p_scale=0.0,
                 p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.ori_size = ori_size

    def get_random_crop_box(self,imgsize, cropsize):
        h, w = imgsize
        ch = min(cropsize, h)
        cw = min(cropsize, w)

        w_space = w - cropsize
        h_space = h - cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space + 1)
            if img_left == 476:
                print(1)
        else:
            cont_left = random.randrange(-w_space + 1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space + 1)
        else:
            cont_top = random.randrange(-h_space + 1)
            img_top = 0

        return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw
    def transform(self,image,to_tensor=True):

        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image = F.to_pil_image(image)



        if np.random.rand() < self.p_hflip:
            image = F.hflip(image)
        if np.random.rand() < self.p_vflip:
            image= F.vflip(image)
        # random rotation
        if np.random.rand() < self.p_rota:
            #
            angle = T.RandomRotation.get_params((-30, 30))
            image= F.rotate(image, angle)
            # angles = [90, 180, 270]
            # index = random.randint(0, 2)
            # angle = angles[index]
            # image =  F.rotate(image, angle)
            # mask = F.rotate(mask, angle)
        # if np.random.rand() < self.p_scale:
        #     scale = np.random.uniform(1, 1.3)
        #     new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
        #     image= F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR)
        #     i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
        #     image = F.crop(image, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            radius = random.random()
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        # if np.random.rand() < self.p_random_affine:
        #     affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
        #     image = F.affine(image, *affine_params)
        # transforming to tensor
        image = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR)
        image = F.to_tensor(image)
        return image
        # if to_tensor:
        #     # to tensor
        #     imgs = [TF.to_tensor(img) for img in imgs]
        #     labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
        #               for img in labels]

        #
        #     # imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        #     #         for img in imgs]
        #     imgs = [TF.normalize(img, mean=[0.5], std=[0.5])
        #             for img in imgs]
        #
        # return imgs, labels
def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value



    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
