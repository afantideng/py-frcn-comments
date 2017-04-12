# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

"""
将经过预处理的 processed_ims 转换成 caffe 支持的数据结构，即 N*C*H*W 的四维结构
输入:装有图片(opencv mat) 的 list (通常只有一张)
输出:可以作为 caffe 输入的 im_blob
"""
def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),  #blob的尺寸:N*H*W*C
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width),转换后blob的尺寸:N*C*H*W
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

# prep_im_for_blob 函数的功能是获取经过resize的图像以及缩放的比例
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)         #目标图片尺寸/当前图片尺寸 得到缩放比例
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,  #把短边归一化为600，长边作相应的缩放, 但不超过1000
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
