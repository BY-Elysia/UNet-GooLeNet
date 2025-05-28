if self.with_scale_random_crop:
    # rescale
    scale_range = [1, 1.2]
    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
    labels = [pil_rescale(img, target_scale, order=0) for img in labels]
    # crop
    imgsize = imgs[0].size  # h, w
    box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
    imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
            for img in imgs]
    labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
              for img in labels]