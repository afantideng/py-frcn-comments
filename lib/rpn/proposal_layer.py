# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    注意：这一层中，始终没有gt信息的直接参与（通过score在nms的时候间接参与）.
    另外：top[1] 的 score 在下一层好像并没有什么用
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]                        #anchor的总个数

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        """
         输出Top[0]: R个roi, 每一个均是 5-tuple (n, x1, y1, x2, y2),
                    其中n 代表batch index； x1, y1, x2, y2表示矩形的4个点的坐标。
　　　　　输出Top[1]: Ｒ个proposal的得分，即是一个物体的可能性。
        """
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # ------只能一张一张地输入------
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N   # TRAIN: 12000 TEST: 6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # TRAIN: 6000  TEST: 300
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH      # TRAIN: 0.7   TEST: 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE        # TRAIN: 16    TEST: 16

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]  # RPN给出的cls_score信息(严格地说是rpn_cls_prob_reshape),前_num_anchors个是背景，之后为前景（物体）

        bbox_deltas = bottom[1].data                          # RPN给出的bbox_pred信息,bbox_deltas就是论文3.1.2节中的't'

        im_info = bottom[2].data[0, :]                        # im_info装的是宽,高和缩放尺度信息

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        # --- rpn_cls_score 这一层输出的 score map 的宽和高 w,h,事实上和　rpn_bbox_pred 输出的宽和高　W,H 相一致　---
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # 通过_feat_stride将最后一层卷积层输出映射回输入的原图尺寸
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride                       # 1*w
        shift_y = np.arange(0, height) * self._feat_stride                      # 1*h
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)                        # h*w
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()      # [(h*w), 4]

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors                                                    # A=9个anchor
        K = shifts.shape[0]                                                      # K=h*w
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))                 # 广播机制,最终结构为 [(h*w*9), 4]
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        """(H*W)×A正好是rpn_bbox_pred层上anchors的总个数,每一个anchor有四个值,又变成(H*W)*A*4"""
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))  #调整bbox_deltas使其与anchors维度相对应:[(h*w*9) ,4]

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))             #调整scores使其与anchors维度相对应:[(h*w*9), 1]
                                                                             #(注意score只取了rpn_cls_score输出一半的fmap)
        """ **** 从所有(h*w*9个)anchors和RPN给出的 bbox_deltas,得到proposals()(论文2.1.3) **** """
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image -- 避免区域越界 --
        proposals = clip_boxes(proposals, im_info[:2])

        """剔除掉proposal中尺寸太小的区域"""
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]        # 留下12000个(TEST:6000个)
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        """---!!!---nms---!!!---"""
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]         # 留下2000个(TEST:300个)
        proposals = proposals[keep, :]
        scores = scores[keep]

        """---RPN只支持输入单张图片---"""
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))  #结构:n*(1+4)
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores                                              #结构:n*1

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
