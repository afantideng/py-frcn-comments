# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    注意：1.这里的评判标准仅有 anchors 和 gt_bbox 的重叠情况（有无物体）,并没有涉及具体的类别
          2.这一层中，rpn_cls_score 仅仅提供了“一个尺寸”，并没有提供卷积得到的特征
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))   #生成anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]                  # rpn_cls_score 输出 map 的宽和高 w,ｈ
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data                                  # gt_bbox (gt_num, 5)
        # im_info
        im_info = bottom[2].data[0, :]                             # im_info (1, 3)

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        """
        ----- 1.生成所有anchors,并且删去越界的anchor -----
        """
        # 1. Generate proposals from bbox deltas and shifted anchors
        # 通过_feat_stride将最后一层卷积层输出映射回输入的原图尺寸
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()      # [(w*h), 4]
        # add A anchors (1, A, 4) to    
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors                                                    # A=9个anchor
        K = shifts.shape[0]                                                      # K=h*w
        all_anchors = (self._anchors.reshape((1, A, 4)) +                        # 产生所有anchors,numpy的广播机制
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))                            # [(h*w*9), 4]
        total_anchors = int(K * A)


        # only keep anchors inside the image
        # 只留下完全在图片中的anchor,这里的 inds_inside
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &   # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)     # height
        )[0] #取要保留下来的anchor的索引值

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]                           # 最终保留下来的anchor [anchors_num, 4]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # 标签 label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)       # 标签的个数等于保留下来的anchor总数 [anchors_num, 1]
        labels.fill(-1)                                                 # 先全部b置为-1



        """
        ---- 2.根据 anchor 和 gt_bbox 的 IOU 打标签 labels (此处首次引入了gt_bbox信息) ----
        """
        # overlaps between the anchors and the gt boxes
        # 计算anchors和gt_boxes之间的overlap（每一个anchor和所有gt之间的overlap）
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(                                             # (anchors_num, gt_bbox_num)
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        #以每一个anchor为标准,寻找与之 overlap 最大的gt_bbox (注意 numpy 的广播机制)
        argmax_overlaps = overlaps.argmax(axis=1)
        #存储IOU
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # 每一个anchor所对应的有最大overlap的gt_bbox
                                                                              # 尺寸: (anchors_num, 1)

        #以每一个gt_bbox为标准,寻找与之 overlap 最大的anchor (注意 numpy 的广播机制)
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        #存储IOU
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]              # 尺寸: (gt_bbox_num, 1)


        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]         # 保证与每一个gt_bbox有‘重叠最大值’的anchor
                                                                              # 都被获取(前面只获取到第一个重叠最大值的anchor位置)
                                                                              # 尺寸：n x 1 (其实是 1 x n)
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0         # 对每一个anchor,最大的 IOU < 0.3 则标为负样本

        # fg label: for each gt, anchor with highest overlap                  # 对每一个gt_bbox,IOU最高的全部anchors都标为正样本
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU                                       # 对每一个anchor,IOU大于0.7的值标为正样本
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0          # 对每一个anchor,IOU<0.3 标为负样本
                                                                               # (再来一次是为了覆盖前两步)

        # subsample positive labels if we have too many                        #下采样以保持正负样本均衡
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many                        #下采样以保持正负样本均衡
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))


        """
        ---- 3.论文3.1.2节中所作的变换，由每一个 anchors 和与之 IOU 最大的 gt_bbox 生成tx,ty,tw,th ----
        """
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)          # 尺寸：(anchors_num, 4)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


        # -- inside weights 联合损失函数中只取正样本的选择系数 [u >= 1]--
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)   # 尺寸：(anchors_num, 4)
        # 正样本为（1,1,1,1) 负样本为 (0,0,0,0)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) 


        # -- outside weights 存放归一化系数 1 / N --
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 尺寸：(anchors_num, 4)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0: # 默认为 -1.0
            # uniform weighting of examples (given non-uniform sampling)
            # 所有为正负样本（标签为0或1）的 anchor 的总个数
            num_examples = np.sum(labels >= 0)
            # 用于归一化的 1 / N
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds



        """
        map up to original set of anchors
        ---- 4.将长度为 len(inds_inside) 的数据映射回长度为 total_anchors.shape[0]的数据，total_anchors = (w*h)*9 ----
        """
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count



        """
        ---- 5.装载 ----
        """
        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)   # (1, A, height, width)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)  # (1, A*4, height, width)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)  # (1, A*4, height, width)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)   # (1, A*4, height, width)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
     size count)
    * count: total_anchors;
    * data: labels, bbox_targets, bbox_inside(outside)_weight
    """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)  # ret尺寸: (total_anchors, 4)
        ret.fill(fill)
        ret[inds, :] = data                                           # 将数据按照原来的地方放回去
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
