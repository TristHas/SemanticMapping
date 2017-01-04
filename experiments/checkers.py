#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from util.preprocessing import get_image, resize, center_crop


def batch_of_syns(wnid = "n02012849", id = 429, batch_size = 256, img_size=224, resize_func = resize):
    synset_dir = os.path.join("/home/tristan/data/ILSVRC2015/Data/CLS-LOC/train", wnid)
    photos = os.listdir(synset_dir)
    input_array = np.zeros((batch_size, 3, img_size, img_size), np.uint8)
    for i, photo in enumerate(photos[:batch_size]):
        print i, photo
        input_array[i,:,:,:] = get_image(os.path.join(synset_dir,photo),
                                               img_size = img_size,
                                               RGB_TO_GBR = True, swapaxing = True)
        #input_array[i,:,:,:] = inception_preprocess(os.path.join(synset_dir,photo),img_size)
    return input_array, batch_size * [id]

def tensor_to_img(tensor, RGB_TO_GBR = True, swapaxing = True):
    if swapaxing:
        tensor = np.rollaxis(tensor, 0, 3)
        if RGB_TO_GBR:
            tensor = tensor[:, :, ::-1]
    else:
        if RGB_TO_GBR:
            tensor = tensor[::-1, :, :]
    return tensor

def show(i, filepath="/home/tristan/data/Imagenet/datasets/ILSVRC2015/256_224/center_crop/valid/input/0"):
    z =  hkl.load(filepath).astype("uint8")
    plt.imshow(tensor_to_img(z[i]))
    plt.show()

