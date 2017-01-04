#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from scipy.misc import imread, imresize
import skimage.transform

import numpy as np


sys.path.append("..")
from util.Helpers import Logger
log = Logger()

def resize(img, img_size = 224):
    """
        Resize the image using scipy.imresize
    """
    target_shape = (img_size, img_size, 3)
    return imresize(img, target_shape)

def center_crop(img, img_size = 224):
    """
        Resize image by setting its lower dimension to img_size then cropping
        the other dimension in its center
    """
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = img.shape
    if h < w:
        img = skimage.transform.resize(img, (img_size, w*img_size/h), preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h*img_size/w, img_size), preserve_range=True)
    # Central crop to img_sizeximg_size
    h, w, _ = img.shape
    img = img[h//2-img_size/2:h//2+img_size/2, w//2-img_size/2:w//2+img_size/2]
    return img

def get_image(file_path, img_size = 224, resize_fn = center_crop,
              RGB_TO_GBR = True, swapaxing = True):
    """
        Opens image given by filepath and apply transform
    """
    img =  imread(file_path)
    log.debug("Getting image {}".format(file_path))
    return transform_image(img, img_size, resize_fn, RGB_TO_GBR, swapaxing)

def transform_image(img, img_size = 224, resize_fn = center_crop,
                    RGB_TO_GBR = True, swapaxing = True):
    """
        Transforms an image given size, channel swapping and a resize method
    """
    # Guarantee three channels
    if len(img.shape) == 2:
        #img =  imresize(img, (img_size, img_size))
        log.debug("img.shape={}".format(img.shape))
        log.debug("Image has dimension 2")
        img = np.expand_dims(img, axis=2)
        img = np.tile(img, (1,1,3))
        log.debug("Image shape: {}".format(img.shape))
    elif img.shape[2] > 3:
        img = img[:, :, :3]
        log.debug("Image has dimension superior to 3")
    # Resize image given the proper resizing method
    img = resize_fn(img, img_size)
    img = image_to_tensor(img, RGB_TO_GBR, swapaxing)
    return img

def image_to_tensor(img, RGB_TO_GBR = True, swapaxing = True):
    """
        Swap dimensions and channels of the input image tensor
    """
    if swapaxing:
        img = np.rollaxis(img, 2)
        if RGB_TO_GBR:
            img = img[::-1, :, :]
    else:
        if RGB_TO_GBR:
            img = img[:, :, ::-1]
    return img

def tensor_to_img(tensor, RGB_TO_GBR = True, swapaxing = True):
    """
        Swap dimensions and channels of the input image tensor
    """
    if swapaxing:
        tensor = np.rollaxis(tensor, 0, 3)
        if RGB_TO_GBR:
            tensor = tensor[:, :, ::-1]
    else:
        if RGB_TO_GBR:
            tensor = tensor[::-1, :, :]
    return tensor


def test_image(file_path):
    try:
        img =  imread(file_path)
        assert len(img.shape) > 1
        return True
    except Exception as e:
        return False
