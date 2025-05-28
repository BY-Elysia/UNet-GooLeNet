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
    def __init__(self, img_size=256,  ori_size=256, crop=False, p_hflip=0.0, p_vflip=0.0,p_rota=0.0, p_scale=0.0,
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
    def transform(self, image, mask, to_tensor=True):
        # """
        # :param imgs: [ndarray,]
        # :param labels: [ndarray,]
        # :return: [ndarray,],[ndarray,]
        # """
        # # resize image and covert to tensor
        # imgs = [TF.to_pil_image(img) for img in imgs]
        # if self.img_size is None:
        #     self.img_size = None
        #
        # if not self.img_size_dynamic:
        #     if imgs[0].size != (self.img_size, self.img_size):
        #         imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
        #                 for img in imgs]
        # else:
        #     self.img_size = imgs[0].size[0]
        #
        # labels = [TF.to_pil_image(img) for img in labels]
        # if len(labels) != 0:
        #     if labels[0].size != (self.img_size, self.img_size):
        #         labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
        #                 for img in labels]
        #
        # random_base = 0.1
        # if self.with_random_hflip and random.random() > random_base:
        #     imgs = [TF.hflip(img) for img in imgs]
        #     labels = [TF.hflip(img) for img in labels]
        #
        # if self.with_random_vflip and random.random() > random_base:
        #     imgs = [TF.vflip(img) for img in imgs]
        #     labels = [TF.vflip(img) for img in labels]
        #
        # if self.with_random_rot and random.random() > random_base:
        #     angles = [90, 180, 270]
        #     index = random.randint(0, 2)
        #     angle = angles[index]
        #     imgs = [TF.rotate(img, angle) for img in imgs]
        #     labels = [TF.rotate(img, angle) for img in labels]
        #
        # if self.with_random_crop and random.random() > 0:
        #     i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
        #         get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))
        #
        #     imgs = [TF.resized_crop(img, i, j, h, w,
        #                             size=(self.img_size, self.img_size),
        #                             interpolation=Image.CUBIC)
        #             for img in imgs]
        #
        #     labels = [TF.resized_crop(img, i, j, h, w,
        #                               size=(self.img_size, self.img_size),
        #                               interpolation=Image.NEAREST)
        #               for img in labels]
        #
        # if self.with_scale_random_crop:
        #     # rescale
        #     scale_range = [1, 1.2]
        #     target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        #
        #     imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
        #     labels = [pil_rescale(img, target_scale, order=0) for img in labels]
        #     # crop
        #     imgsize = imgs[0].size  # h, w
        #     box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
        #     imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
        #             for img in imgs]
        #     labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
        #             for img in labels]
        #
        # if self.with_random_blur and random.random() > 0:
        #     radius = random.random()
        #     imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
        #             for img in imgs]
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)



        if np.random.rand() < self.p_hflip:
            image, mask = F.hflip(image), F.hflip(mask)
        if np.random.rand() < self.p_vflip:
            image, mask = F.vflip(image), F.vflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            #
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
            # angles = [90, 180, 270]
            # index = random.randint(0, 2)
            # angle = angles[index]
            # image =  F.rotate(image, angle)
            # mask = F.rotate(mask, angle)
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w),
                                                                                                InterpolationMode.NEAREST)
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
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
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (
            self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        # random crop
        # if self.crop:
        #     scale_range = [1, 1.2]
        #     target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        #
        #     image = pil_rescale(image, target_scale, order=3)
        #     mask = pil_rescale(mask, target_scale, order=0)
        #     # crop
        #     imgsize = image.size  # h, w
        #     box = self.get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
        #     image = pil_crop(image, box, cropsize=self.img_size, default_value=0)
        #
        #     mask = pil_crop(mask, box, cropsize=self.img_size, default_value=255)
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)
        # image = F.normalize(image, mean=[0.5, 0.5,0.5],std=[0.5, 0.5,0.5])
        return image, mask
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
