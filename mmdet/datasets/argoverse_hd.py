# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ArgoverseDataset(CocoDataset):

    CLASSES = ('person',  'bicycle',  'car',  'motorcycle',  'bus',  'truck',  
                'traffic_light',  'stop_sign')
    
    PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
               (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
               (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
               (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
               (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        sequences = self.coco.dataset['sequences']  # hc-y_add1031:
        seq_dirs = self.coco.dataset['seq_dirs']  # hc-y_add1031:
        data_infos = []
        total_ann_ids = []
        # hc-y_note1231:只采样出少量的图片用于训练和评估;
        # if 'train.json' in ann_file:
        #     self.img_ids = self.img_ids[::8]  # './../datasets/Argoverse-1.1/annotations/train.json' 39384 images
        # elif 'val.json' in ann_file:
        #     self.img_ids = self.img_ids[::5]  # './../datasets/Argoverse-1.1/annotations/val.json' 15062 images
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = seq_dirs[info['sid']] + '/' + info['name']  # hc-y_modify1031:
            # from pathlib import Path  # hc-y_add1031:
            # Path('/mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-1.1/images/' + seq_dirs[info['sid']]+'/'+info['name']).exists()  # hc-y_add1031:
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
