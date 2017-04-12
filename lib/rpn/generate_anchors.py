# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # 生成单个 base_scale 时同一组 ratios 下的所有anchors
    # 尺寸: (ratio_nums, 4)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # 计算每一个 scale 下的 anchors 得到所有 anchors
    # 尺寸: (ratio_nums x scale_nums, 4)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors


"""
 返回anchor的宽,高和中心坐标
 **被_ratio_enum和_scale_enum调用**
"""
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

"""
 给定一组对应的 w 和 h,生成所有 anchors (左上和右下两个点的坐标)
 输出尺寸：(w_or_h_nums, 4)
 注意：生成的 anchors 均以原点为中心
 **被_ratio_enum和_scale_enum调用**
"""
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

"""
 根据一个基准的anchor和给定的ratios,生成所有符合条件的anchors
"""
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each 'aspect ratio' wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    # 所有宽度值的向量 (1, ratio_nums)
    ws = np.round(np.sqrt(size_ratios))
    # 所有高度值的向量 (1, ratio_nums)
    hs = np.round(ws * ratios)
    # 所有anchors (ratio_nums, 4)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

"""
 根据一个基准的anchor和给定的scales,生成所有符合条件的anchors
"""
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each 'scale' wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # 所有宽度值的向量 (1, scale_nums)
    ws = w * scales
    # 所有高度值的向量 (1, scale_nums)
    hs = h * scales
    # 所有anchors (scale_nums, 4)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
