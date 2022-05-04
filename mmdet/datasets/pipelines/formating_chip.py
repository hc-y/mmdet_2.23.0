# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import to_tensor, ImageToTensor, DefaultFormatBundle


@PIPELINES.register_module()
class ImageToTensorChipsV1v1(ImageToTensor):  # hc-y_add0105:
    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        if 'chip4' in results:
            chip4_list = []
            for _chip in results['chip4']:
                if len(_chip.shape) < 3:
                    _chip = np.expand_dims(_chip, -1)
                chip4_list.append((to_tensor(_chip.transpose(2, 0, 1))).contiguous())
            results['chip4'] = torch.stack(chip4_list, 0)
        return results


@PIPELINES.register_module()
class ImageToTensorChipsV1v2(ImageToTensor):  # hc-y_add0106:
    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        if 'chip4' in results:
            chip4_list = []
            for _chip in results['chip4']:
                if len(_chip.shape) < 3:
                    _chip = np.expand_dims(_chip, -1)
                chip4_list.append((to_tensor(_chip.transpose(2, 0, 1))).contiguous())
            results['chip4'] = torch.stack(chip4_list, 0)
        if 'chip3_fts_idx' in results:
            results['chip3_fts_idx'] = to_tensor(np.stack(results['chip3_fts_idx'], axis=0))
        return results


@PIPELINES.register_module()
class DefaultFormatBundleChipsV1v1(DefaultFormatBundle):  # hc-y_add0105:
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        if 'chip4' in results:
            chip4_list = []
            for _chip in results['chip4']:
                if self.img_to_float is True and _chip.dtype == np.uint8:
                    _chip = _chip.astype(np.float32)
                if len(_chip.shape) < 3:
                    _chip = np.expand_dims(_chip, -1)
                chip4_list.append(np.ascontiguousarray(_chip.transpose(2, 0, 1)))
            results['chip4'] = DC(to_tensor(np.stack(chip4_list, axis=0)), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'], 
                padding_value=self.pad_val['masks'], 
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), 
                padding_value=self.pad_val['seg'], 
                stack=True)
        return results


@PIPELINES.register_module()
class DefaultFormatBundleChipsV1v2(DefaultFormatBundle):  # hc-y_add0106:
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        if 'chip4' in results:
            chip4_list = []
            for _chip in results['chip4']:
                if self.img_to_float is True and _chip.dtype == np.uint8:
                    _chip = _chip.astype(np.float32)
                if len(_chip.shape) < 3:
                    _chip = np.expand_dims(_chip, -1)
                chip4_list.append(np.ascontiguousarray(_chip.transpose(2, 0, 1)))
            results['chip4'] = DC(to_tensor(np.stack(chip4_list, axis=0)))
        if 'chip3_fts_idx' in results:
            results['chip3_fts_idx'] = DC(to_tensor(np.stack(results['chip3_fts_idx'], axis=0)))
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'], 
                padding_value=self.pad_val['masks'], 
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), 
                padding_value=self.pad_val['seg'], 
                stack=True)
        return results

@PIPELINES.register_module()
class DefaultFormatBundleChipsV1v3(DefaultFormatBundle):  # hc-y_add0502:
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))

            chips_list = [img,]
            for chip_fields in results.get('chips_fields', []):
                _chip = chip_fields.pop('cimg')
                if self.img_to_float is True and _chip.dtype == np.uint8:
                    _chip = _chip.astype(np.float32)
                if len(_chip.shape) < 3:
                    _chip = np.expand_dims(_chip, -1)
                chips_list.append(np.ascontiguousarray(_chip.transpose(2, 0, 1)))

            results['img'] = DC(to_tensor(chips_list), stack=False)  # hc-y_note0502: num_chips of each image may be different, so set stack=False here;
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'], 
                padding_value=self.pad_val['masks'], 
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), 
                padding_value=self.pad_val['seg'], 
                stack=True)
        return results
