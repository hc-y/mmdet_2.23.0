# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
# from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
from .loading import LoadAnnotations

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadAnnotationsWChipsV1(LoadAnnotations):  # hc-y_add0430:
    def __init__(self,
                *args,
                 with_chip=True,
                 **kwargs):
        super(LoadAnnotationsWChipsV1, self).__init__(*args, **kwargs)
        self.with_chip = with_chip

    def _crop_chips(self, results):
        """Private function to crop chips from original image, 
            and generate bounding box,label annotations for each chip.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains chips with bounding box,label annotations.
        """
        # lf: last frame; cf: current frame; nf: next frame;
        # chips_lf = np.array(results['img_info']['chips_lft'][0], dtype=np.float32)
        # hc-y_TODO: 给 chips_lf 施加一个随机抖动
        chips_lf = results['img_info']['chips_lft'][0]
        
        chips_img = []
        chips_params = []
        for i in range(len(chips_lf)):
            x1a0,y1a0,x2a0,y2a0 = (int(_val) for _val in chips_lf[0])
            chips_img.append(results['img'][y1a0:y2a0, x1a0:x2a0])
            gt_bboxes_clipped = results['gt_bboxes'].copy()
            np.clip(gt_bboxes_clipped[:,0::2], x1a0, x2a0, out=gt_bboxes_clipped[:,0::2])
            np.clip(gt_bboxes_clipped[:,1::2], y1a0, y2a0, out=gt_bboxes_clipped[:,1::2])
            ious_itself = bbox_overlaps(gt_bboxes_clipped, results['gt_bboxes'], mode='iou', is_aligned=True)
            idx_rest = np.where(ious_itself >= 0.5)[0]  # 当gt bbox被clip掉大部分比例时, 直接删除该gt bbox;
            gt_bboxes_clipped[:, 0::2] -= x1a0
            gt_bboxes_clipped[:, 1::2] -= y1a0
            chips_params.append([(x1a0,y1a0,x2a0,y2a0), gt_bboxes_clipped[idx_rest], results['gt_labels'][idx_rest]])
        results['chips_img'] = chips_img
        results['chips_params'] = chips_params
        return results


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_chip:
            results = self._crop_chips(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_chip={self.with_chip}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str
