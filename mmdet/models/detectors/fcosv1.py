# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms
from mmdet.core import bbox2result, bbox_overlaps
from mmdet.utils.general import xyxy2xywh
from ..builder import DETECTORS
from .fcos import FCOS


@DETECTORS.register_module()
class FCOSV0v1(FCOS):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""
    # hc-y_add0529:def merge_chips_result(), def simple_test(), def forward_test() 均复制自 mmdet/models/detectors/yoloxv1.py, 一模一样

    def merge_chips_result(self, result_list_per_img, img_meta):  # hc-y_add0502:
        if len(result_list_per_img) == 1:
            return result_list_per_img[0]
        else:
            result0_per_img = result_list_per_img.pop(0)
            chips_ltrb = [chip_field['c_ltrb'] for chip_field in img_meta['chips_fields']]
            for i_c in range(len(chips_ltrb)):
                result_list_per_img[i_c][0][:, [0,2]] += chips_ltrb[i_c][0]
                result_list_per_img[i_c][0][:, [1,3]] += chips_ltrb[i_c][1]
            bbox_per_img, label_per_img = [result0_per_img[0],], [result0_per_img[1],]
            flag_with_ibs = False
            # flag_with_ibs = True
            if flag_with_ibs:
                for i_c in range(len(chips_ltrb)):
                    if len(result_list_per_img[i_c][0]) == 0:
                        continue
                    bbox_pred_curchip = result_list_per_img[i_c][0]
                    bbox_pred_otherchip = torch.cat([result0_per_img[0],] + [result_list_per_img[j][0] for j in range(len(chips_ltrb)) if j != i_c and len(result_list_per_img[j][0]) > 0],0)
                    chip_ltrb = torch.tensor([chips_ltrb[i_c]]).to(bbox_pred_curchip.device)
                    # mask_keep = ((bbox_pred_curchip[:, :4] - chip_ltrb).abs() *2 / (chip_ltrb[:, 2:] - chip_ltrb[:,:2]).repeat(1,2)).min(dim=1)[0] < xxx
                    mask_keep = (bbox_pred_curchip[:, :4] - chip_ltrb).abs().min(dim=1)[0] >= 8  # hc-y_note0602:此行没有写入论文
                    iofs_otherchip, bbox_in = bbox_overlaps(bbox_pred_otherchip[:, :4], chip_ltrb, mode='iof', is_aligned=False, bbox_in=True)
                    mask_truncated = (iofs_otherchip[:, 0] > 0) & (iofs_otherchip[:, 0] < 1.) & (bbox_pred_otherchip[:, 4] > self.test_cfg.score_thr)  # torch.logical_and(iofs_otherchip[:, 0] > 0., iofs_otherchip[:, 0] < 1.)
                    if mask_truncated.sum() > 0:
                        ious_curchip = bbox_overlaps(bbox_pred_curchip[:, :4], bbox_in[mask_truncated, 0], mode='iou', is_aligned=False)
                        # hc-y_note0602:如果没加 & mask_keep,此方法的局限性在于:如果截断目标在其它patch中没有被检测到, 那么就没法被移除掉; 
                        mask_keep = (ious_curchip.max(dim=1)[0] < 0.9) & mask_keep  # 超参数, 待调节;
                    bbox_per_img.append(result_list_per_img[i_c][0][mask_keep])
                    label_per_img.append(result_list_per_img[i_c][1][mask_keep])
            else:
                for i_c in range(len(chips_ltrb)):
                    if len(result_list_per_img[i_c][0]) == 0:
                        continue
                    bbox_per_img.append(result_list_per_img[i_c][0])
                    label_per_img.append(result_list_per_img[i_c][1])
            bbox_per_img = torch.cat(bbox_per_img, 0)
            label_per_img = torch.cat(label_per_img, 0)
            # return bbox_per_img, label_per_img
            cfg_nms = self.test_cfg.nms.copy()
            # cfg_nms['iou_threshold'] = 0.75
            dets, keep = batched_nms(bbox_per_img[:, :4], bbox_per_img[:, 4], label_per_img, cfg_nms)

            if False:
            # if True:
                from pathlib import Path
                path_to_tmp = Path('./my_workspace/tmp/')
                path_to_img = img_meta['filename']
                img_hw = img_meta['ori_shape'][:2]
                from mmdet.utils import plot_images_v1
                from mmdet.utils import xyxy2xywhn
                import numpy as np
                bbox_vis, label_vis = result0_per_img[0], result0_per_img[1]
                # i_c = 0
                # bbox_vis, label_vis = result_list_per_img[i_c][0], result_list_per_img[i_c][1]
                # bbox_vis, label_vis = bbox_per_img, label_per_img
                # bbox_vis, label_vis = dets, label_per_img[keep]
                bbox_vis_ = xyxy2xywhn(bbox_vis[:, :4], w=img_hw[1], h=img_hw[0])
                l_dets = torch.cat((torch.zeros_like(label_vis[:,None]), label_vis[:,None], bbox_vis_, bbox_vis[:,4:5]), dim=1)
                cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
                plot_images_v1(None, l_dets.cpu().numpy(), (path_to_img, ), path_to_tmp / f'img_c.jpg', cls_names, None, 'original_image')
            return dets, label_per_img[keep]

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (List[torch.Tensor]): the outer list corresponds to N images  # hc-y_modify0502:
                in the batch, the inner Tensor should have a shape num_chipxCxHxW, 
                which indicates each image along with its chips.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        results_list = []  # hc-y_modify0502:
        for img_id, _img in enumerate(img):
            feat = self.extract_feat(_img)
            result_list_per_img = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale)
            results_list.append(self.merge_chips_result(result_list_per_img, img_metas[img_id]))
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results, results_list  # hc-y_modify0501:原为 bbox_results
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[List[Tensor]]): the outer list indicates test-time  # hc-y_modify0502:
                augmentations and the inner list corresponds to N images 
                in the batch, the inner Tensor should have a shape num_chipxCxHxW, 
                which indicates each image along with its chips.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img[0].size()[-2:])  # hc-y_modify0502:

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
