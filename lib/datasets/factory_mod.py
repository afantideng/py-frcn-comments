# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc_mod import pascal_voc
import numpy as np

name='VOC2007'
devkit_path='/home/carrot/DLnetworks/faster-rcnn-VOC/data/VOCdevkit2007'
for image_set in ['train','val','test']: 
    full_name=name+'_'+image_set
    __sets[full_name]=(lambda img_set=image_set,devkit=devkit_path:pascal_voc(img_set,devkit))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
