# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
from .transforms import Resize, Normalize

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class ResizeChipsV1v1(Resize):  # hc-y_add0104:
    """hc-y_add0104:将原始图片均匀切分成4块并 resize 到指定大小;"""
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            # hc-y_add0104:需确保 img_sz 是32的整数倍, chip_sz 也是32的整数倍, 从而使得 img, chip 无需 Pad 就可以作为 model input;
            h0, w0 = results[key].shape[:2]
            chip4_list, labels4_list, chip3_fts_idx = [], [], []
            h0_half, w0_half = int(h0 / 2), int(w0 / 2)  # 注意: w0_half * 2 和 h0_half * 2 分别都有可能小于 w0和h0;
            for j in range(4):
                if j == 0:  # top left
                    x1a, y1a, x2a, y2a = 0, 0, w0_half, h0_half
                elif j == 1:  # top right
                    x1a, y1a, x2a, y2a = w0_half, 0, w0_half * 2, h0_half
                elif j == 2:  # bottom left
                    x1a, y1a, x2a, y2a = 0, h0_half, w0_half, h0_half * 2
                elif j == 3:  # bottom right
                    x1a, y1a, x2a, y2a = w0_half, h0_half, w0_half * 2, h0_half * 2
                    assert x2a == w and y2a == h
                chip4_list.append(cv2.resize(results[key][y1a:y2a, x1a:x2a], (1024, 640), interpolation=cv2.INTER_LINEAR))
            results['chip4'] = chip4_list
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class ResizeChipsV1v2(Resize):  # hc-y_add0105:
    """hc-y_add0105:crop chips with different FoV, rescale and append to list;"""
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            # hc-y_add0104:需确保 img_sz 是32的整数倍, chip_sz 也是32的整数倍, 从而使得 img, chip 无需 Pad 就可以作为 model input;
            img0 = results[key]
            gt_anns = np.concatenate((results['gt_labels'][:,None], results['gt_bboxes']), 1) if len(results.get('gt_labels', [])) else None # (category,x1,y1,x2,y2)
            hw_unpad, hw_pad, pad_ltrb = self.letterbox_pad_params(img.shape[:2], divisor=32)
            # self.calculate_chip_xywh_after(results[key].shape[:2], hw_unpad, pad_ltrb, (h_scale, w_scale), ftsfus_stride_max=32)  # hc-y_add0114:
            ftsfus_stride_max = 32
            chips_xywh = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
            offsets_topleft = np.array([[1, 0], [1, 0], [1, 0]])
            # append each chip to chip4_list; update labels in each chip to labels4_list;
            chip4_list, labels4_list, chip3_fts_idx = [], [], []
            flag_wo_ftsftu = False
            if flag_wo_ftsftu:
                chip4_list.append(img)
                chip3_fts_idx.append(None)
            else:
                # from tools.general import increment_path, xyxy2xywhn
                # path_to_tmp = increment_path('/mnt/data2/yuhangcheng/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server58
                # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
                # cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
                # from tools.plots import plot_images_v1
                for j in range(4):
                    labels = None if gt_anns is None else gt_anns.copy()
                    if j == 0:
                        chip = img
                        if gt_anns is not None:
                            # hc-y_note0106:参考 def _resize_bboxes(self, results)
                            labels[:, 1:] = labels[:, 1:] * scale_factor
                            if self.bbox_clip_border:
                                labels[:, 1::2] = np.clip(labels[:, 1::2], 0, hw_unpad[1])
                                labels[:, 2::2] = np.clip(labels[:, 2::2], 0, hw_unpad[0])
                    elif j > 0:
                        # (x1aa,y1aa,waa,haa)表示chip在model input with the full-global FoV中的像素索引(left,top,width,height); 此处约束了chip在feature map of full-image中的特征索引为整数;
                        x1aa = math.floor((hw_unpad[1] * (chips_xywh[j-1,0] - chips_xywh[j-1,2] / 2) + pad_ltrb[0]) / ftsfus_stride_max + offsets_topleft[j-1,1]) * ftsfus_stride_max
                        y1aa = math.floor((hw_unpad[0] * (chips_xywh[j-1,1] - chips_xywh[j-1,3] / 2) + pad_ltrb[1]) / ftsfus_stride_max + offsets_topleft[j-1,0]) * ftsfus_stride_max
                        waa = math.floor(hw_unpad[1] * chips_xywh[j-1,2] / ftsfus_stride_max) * ftsfus_stride_max
                        haa = math.floor(hw_unpad[0] * chips_xywh[j-1,3] / ftsfus_stride_max) * ftsfus_stride_max

                        # (x1a0,y1a0,wa0,ha0)表示chip在原始高分辨率图片中的像素索引(left,top,width,height)
                        x1a0, wa0 = (int(_val / w_scale) for _val in (x1aa - pad_ltrb[0], waa))
                        y1a0, ha0 = (int(_val / h_scale) for _val in (y1aa - pad_ltrb[1], haa))

                        h0_chip_canvas, w0_chip_canvas = (math.floor(_val / ftsfus_stride_max) * ftsfus_stride_max for _val in hw_pad)
                        ratio_chip_canvas = min(w0_chip_canvas/wa0, h0_chip_canvas/ha0)
                        # (0,0,wbb,hbb)表示chip在model input with the local FoV中的像素索引
                        wbb, hbb = (round(_val * ratio_chip_canvas) for _val in (wa0, ha0))
                        assert wbb%ftsfus_stride_max == 0 or hbb%ftsfus_stride_max == 0
                        # hw_pad = (hbb, wbb)  # 注释此行时, chip4_list 中的各个 chip 通过 padding 而保证宽高一致;

                        chip3_fts_idx.append(np.array(((x1aa, y1aa, waa, haa), (0, 0, wbb, hbb))))

                        chip = np.full((hw_pad[0], hw_pad[1], img.shape[2]), 114, dtype=np.uint8)  # padded chip as one of the model inputs
                        # place the chip cropped from original image
                        chip[:hbb, :wbb] = cv2.resize(img0[y1a0:(y1a0+ha0), x1a0:(x1a0+wa0)], (wbb, hbb), interpolation=cv2.INTER_LINEAR)

                        if gt_anns is not None:
                            gt_bbox_wh_before_clip = np.stack((labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]), 1)
                            np.clip(labels[:, 1::2], x1a0, x1a0+wa0, out=labels[:, 1::2])
                            np.clip(labels[:, 2::2], y1a0, y1a0+ha0, out=labels[:, 2::2])
                            gt_bbox_wh_after_clip = np.stack((labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]), 1)
                            idx_del = np.where(np.amin(gt_bbox_wh_after_clip / gt_bbox_wh_before_clip, axis=1) < 1/3)[0]  # 当gt bbox被clip掉大部分比例时, 直接删除该gt bbox;
                            labels = np.delete(labels, idx_del, axis=0)
                            labels[:, 1::2] = labels[:, 1::2] - x1a0
                            labels[:, 2::2] = labels[:, 2::2] - y1a0
                            labels[:, 1:] = labels[:, 1:] * ratio_chip_canvas
                    # cv2.imwrite(str(path_to_tmp / f'img_chip{j}.jpg'), chip)  # save
                    # labels[:, 1:] = xyxy2xywhn(labels[:, 1:], hw_pad[1], hw_pad[0])
                    # plot_images_v1(None, np.concatenate((np.zeros_like(labels[:,0:1]), labels), -1), (str(path_to_tmp / f'img_chip{j}.jpg'), ), path_to_tmp / f'img_chip{j}_gts.jpg', cls_names, None, 'original_image')
                    # print('')

                    chip4_list.append(chip)
                    labels4_list.append(labels)
            results['chip4'] = chip4_list
            results['chip3_fts_idx'] = chip3_fts_idx
            results[key] = img

            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def letterbox_pad_params(self, shape_resized, divisor=32):
        """
        hc-y_note0105:注意确保 padding 方式 与 mmdet_1213/mmdet/datasets/pipelines/transforms.py def _pad_img() --> mmcv.impad_to_multiple() 一致;
        Args:
            shape_resized (Tuple): (height, width);
        Returns:
            shape_pad (Tuple): (height, width);
            padding_ltrb (Tuple): the padding for the left, top, right and bottom borders respectively;
        """
        pad_h = int(np.ceil(shape_resized[0] / divisor)) * divisor
        pad_w = int(np.ceil(shape_resized[1] / divisor)) * divisor
        shape_pad = (pad_h, pad_w)
        # left, top, right, bottom, 
        padding_ltrb = (0, 0, shape_pad[1] - shape_resized[1], shape_pad[0] - shape_resized[0])
        
        return shape_resized, shape_pad, padding_ltrb

    def calculate_chip_xywh_after(self, hw_img0, hw_unpad, pad_ltrb, hw_scale, ftsfus_stride_max=32):  # hc-y_add0114:
        # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1280, 800), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.52, 0.5, 0.48) <-- (0.5, 0.5, 0.5, 0.5)
        # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1024, 640), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.55, 0.5, 0.5) <-- (0.5, 0.5, 0.5, 0.5)
        chips_xywh = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.8]])
        offsets_topleft = np.array([[1, 0], [1, 0], [1, 0]])
        h_scale, w_scale = hw_scale[0], hw_scale[1]
        for j in range(1, 4):
            x1aa = math.floor((hw_unpad[1] * (chips_xywh[j-1,0] - chips_xywh[j-1,2] / 2) + pad_ltrb[0]) / ftsfus_stride_max + offsets_topleft[j-1,1]) * ftsfus_stride_max
            y1aa = math.floor((hw_unpad[0] * (chips_xywh[j-1,1] - chips_xywh[j-1,3] / 2) + pad_ltrb[1]) / ftsfus_stride_max + offsets_topleft[j-1,0]) * ftsfus_stride_max
            waa = math.floor(hw_unpad[1] * chips_xywh[j-1,2] / ftsfus_stride_max) * ftsfus_stride_max
            haa = math.floor(hw_unpad[0] * chips_xywh[j-1,3] / ftsfus_stride_max) * ftsfus_stride_max

            x1a0, wa0 = (int(_val / w_scale) for _val in (x1aa - pad_ltrb[0], waa))
            y1a0, ha0 = (int(_val / h_scale) for _val in (y1aa - pad_ltrb[1], haa))
            _chip_xywh_2 = np.array([(x1a0 + wa0/2) / hw_img0[1], (y1a0 + ha0/2) / hw_img0[0], wa0 / hw_img0[1], ha0 / hw_img0[0]])
            print(f'imgsz={(hw_img0[1], hw_img0[0])}, inputsz={(hw_unpad[1], hw_unpad[0])}, offset_topleft={offsets_topleft[j-1]}时, chip_xywh = {_chip_xywh_2} <-- {chips_xywh[j-1]}')
            print(f'chip_xywh before round: {chips_xywh[j-1]}\nchip_xywh after round: {_chip_xywh_2}\n')


@PIPELINES.register_module()
class NormalizeChipsV1v1(Normalize):
    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        if 'chip4' in results:
            for j in range(len(results['chip4'])):
                results['chip4'][j] = mmcv.imnormalize(results['chip4'][j], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results
