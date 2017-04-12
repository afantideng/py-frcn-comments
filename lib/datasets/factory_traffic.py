# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.traffic_sign import traffic_sign
import numpy as np

name='traffic_sign'
devkit_path='/home/carrot/DATA/TrafficSign/AnnoData'
for image_set in ['train','val']:
    full_name=name+'_'+image_set
    __sets[full_name]=(lambda img_set=image_set,devkit=devkit_path:traffic_sign(img_set,devkit))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
