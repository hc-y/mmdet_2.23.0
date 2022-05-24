# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import DETECTORS, build_neck
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result


@DETECTORS.register_module()
class YOLOFV0v1(SingleStageDetector):  # hc-y_add0109:
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 ftsfusneck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOFV0v1, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
        self.ftsfusneck = build_neck(ftsfusneck)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)  # torch.Size([bs, c, h, w])
        x = self.ftsfusneck(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        feat = self.ftsfusneck(feat)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


@DETECTORS.register_module()  # hc-y_add0104:
class YOLOFMBranchV1v1(SingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 local_branch=dict(),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOFMBranchV1v1, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
        self.local_branch = local_branch

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None, chip4=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
            chip4 (Tensor): Tensor with shape torch.Size([bs, num_chip, 3, 640, 1024]);

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        # from mmdet.utils import increment_path, mmdet_imdenormalize
        # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
        # import cv2
        # cv2.imwrite(str(path_to_tmp / f'img0_resized.jpg'), mmdet_imdenormalize(img[0].cpu().numpy().transpose((1, 2, 0))))
        if self.local_branch.get('with_no_grad', True):
            with torch.no_grad():
                chip4_list = []
                for j in range(chip4.size(1)):  # chip4: torch.Size([bs, num_chip, 3, 640, 1024])
                    chip4_list.append(self.extract_feat(chip4[:, j])[0])  # torch.Size([bs, c, h', w'])
                    # chip4_list.append(chip4[:, j])
                    # cv2.imwrite(str(path_to_tmp / f'img0_chip{j}.jpg'), mmdet_imdenormalize(chip4[0, j].cpu().numpy().transpose((1, 2, 0))))
                _chip4_tltr = torch.cat((chip4_list[0], chip4_list[1]), dim=3)
                _chip4_blbr = torch.cat((chip4_list[2], chip4_list[3]), dim=3)
                _chip4 = torch.cat((_chip4_tltr, _chip4_blbr), dim=2)  # torch.Size([bs, c, h'*2, w'*2])
                # cv2.imwrite(str(path_to_tmp / f'img0_chips_ori.jpg'), mmdet_imdenormalize(_chip4[0].cpu().numpy().transpose((1, 2, 0))))
        else:
            chip4_list = []
            for j in range(chip4.size(1)):  # chip4: torch.Size([bs, num_chip, 3, 640, 1024])
                chip4_list.append(self.extract_feat(chip4[:, j])[0])  # torch.Size([bs, c, h', w'])
            _chip4_tltr = torch.cat((chip4_list[0], chip4_list[1]), dim=3)
            _chip4_blbr = torch.cat((chip4_list[2], chip4_list[3]), dim=3)
            _chip4 = torch.cat((_chip4_tltr, _chip4_blbr), dim=2)  # torch.Size([bs, c, h'*2, w'*2])
        
        x = self.extract_feat(img)  # torch.Size([bs, c, h, w])
        # x = (img, )

        bs, _dst_hw = x[0].shape[0], x[0].shape[-2:]
        _src_hw, _src_patch_x1y1wh = _chip4.shape[-2:], torch.tensor([0., 0., _chip4.size(-1), _chip4.size(-2)], device=x[0].device)
        # _src_patch_x1y1wh = torch.tensor([_src_hw[1]/2-_src_hw[1]*0.4/2, _src_hw[0]/2-_src_hw[0]*0.4/2, _src_hw[1]*0.4, _src_hw[0]*0.4], device=x[0].device)
        _yy = (torch.arange(0, _dst_hw[0], device=x[0].device).to(x[0].dtype) + 0.5) * _src_patch_x1y1wh[3] / _dst_hw[0] + _src_patch_x1y1wh[1]
        _xx = (torch.arange(0, _dst_hw[1], device=x[0].device).to(x[0].dtype) + 0.5) * _src_patch_x1y1wh[2] / _dst_hw[1] + _src_patch_x1y1wh[0]
        _grid_y = (_yy / _src_hw[0] * 2 - 1)[None, :, None].expand(bs, _dst_hw[0], _dst_hw[1])
        _grid_x = (_xx / _src_hw[1] * 2 - 1)[None, None, :].expand(bs, _dst_hw[0], _dst_hw[1])
        _chip4_resized = F.grid_sample(_chip4, torch.stack([_grid_x, _grid_y], dim=3), align_corners=False)  # torch.Size([bs, c, h, w])
        # cv2.imwrite(str(path_to_tmp / f'img0_chips.jpg')+, mmdet_imdenormalize(_chip4_resized[0].cpu().numpy().transpose((1, 2, 0))))

        x = (torch.cat((x[0],_chip4_resized), dim=1),)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses


    def simple_test(self, img, img_metas, rescale=False, chip4=None):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        if self.local_branch.get('with_no_grad', True):
            with torch.no_grad():
                chip4_list = []
                for j in range(chip4.size(1)):  # chip4: torch.Size([bs, num_chip, 3, 640, 1024])
                    chip4_list.append(self.extract_feat(chip4[:, j])[0])  # torch.Size([bs, c, h', w'])
                _chip4_tltr = torch.cat((chip4_list[0], chip4_list[1]), dim=3)
                _chip4_blbr = torch.cat((chip4_list[2], chip4_list[3]), dim=3)
                _chip4 = torch.cat((_chip4_tltr, _chip4_blbr), dim=2)  # torch.Size([bs, c, h'*2, w'*2])
        else:
            chip4_list = []
            for j in range(chip4.size(1)):  # chip4: torch.Size([bs, num_chip, 3, 640, 1024])
                chip4_list.append(self.extract_feat(chip4[:, j])[0])  # torch.Size([bs, c, h', w'])
            _chip4_tltr = torch.cat((chip4_list[0], chip4_list[1]), dim=3)
            _chip4_blbr = torch.cat((chip4_list[2], chip4_list[3]), dim=3)
            _chip4 = torch.cat((_chip4_tltr, _chip4_blbr), dim=2)  # torch.Size([bs, c, h'*2, w'*2])
        
        x = self.extract_feat(img)

        bs, _dst_hw = x[0].shape[0], x[0].shape[-2:]
        _src_x1y1wh = torch.tensor([0., 0., _chip4.size(-1), _chip4.size(-2)], device=x[0].device)
        _yy = torch.arange(0, _dst_hw[0], device=x[0].device).to(x[0].dtype) + 0.5
        _xx = torch.arange(0, _dst_hw[1], device=x[0].device).to(x[0].dtype) + 0.5
        _grid_y = ((_yy - _src_x1y1wh[1]) / _dst_hw[0] * 2 - 1)[None, :, None].expand(bs, _dst_hw[0], _dst_hw[1])
        _grid_x = ((_xx - _src_x1y1wh[0]) / _dst_hw[1] * 2 - 1)[None, None, :].expand(bs, _dst_hw[0], _dst_hw[1])
        _chip4_resized = F.grid_sample(_chip4, torch.stack([_grid_x, _grid_y], dim=3), align_corners=False)  # torch.Size([bs, c, h, w])

        feat = (torch.cat((x[0],_chip4_resized), dim=1),)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
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
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            if 'chip4' in kwargs:  # hc-y_add0108:
                kwargs['chip4'] = kwargs['chip4'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)


@DETECTORS.register_module()
class YOLOFMBranchV1v2(SingleStageDetector):  # hc-y_add0106:
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 ftsfusneck,
                 bbox_head,
                 local_branch=dict(),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOFMBranchV1v2, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
        self.ftsfusneck = build_neck(ftsfusneck)
        self.local_branch = local_branch

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None, chip4=None, chip3_fts_idx=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
            chip4 (List(Tensor)): each Tensor with shape torch.Size([num_chips, 3, chip_h, chip_w]);
            chip3_fts_idx (List(Tensor)): outer list 对应于 each img, each Tensor with shape torch.Size([num_chips, 2, 4]); for each chip, 由(x1aa, y1aa, waa, haa), (0, 0, wbb, hbb)构成;

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        idx_chip = 2  # 选取哪几个chip的特征用于feature fusion, 这里只选取了第2个chip
        chip4 = torch.stack(chip4, 0)
        chip3_fts_idx = torch.stack(chip3_fts_idx, 0)
        if self.local_branch.get('with_no_grad', True):
            with torch.no_grad():
                fts_fovs = self.extract_feat(chip4[:, idx_chip])[0]  # torch.Size([bs, c, h', w'])
                # fts_fovs = chip4[:, idx_chip]
        else:
            fts_fovs = self.extract_feat(chip4[:, idx_chip])[0]  # torch.Size([bs, c, h', w'])
        fts_fovl = self.extract_feat(img)[0]  # torch.Size([bs, c, h, w])
        # fts_fovl = img

        fts_stride = img.size(-2) / fts_fovl.size(-2)  # fts_stride (float): 8.0/16.0/32.0;
        chip_fts_idx = chip3_fts_idx[:, idx_chip-1] / fts_stride
        x = self.ftsfusneck([fts_fovl, fts_fovs], chip_fts_idx)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, chip4=None, chip3_fts_idx=None):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        idx_chip = 2  # 选取哪几个chip的特征用于feature fusion, 这里只选取了第2个chip
        if self.local_branch.get('with_no_grad', True):
            with torch.no_grad():
                fts_fovs = self.extract_feat(chip4[:, idx_chip])[0]  # torch.Size([bs, c, h', w'])
        else:
            fts_fovs = self.extract_feat(chip4[:, idx_chip])[0]  # torch.Size([bs, c, h', w'])
        fts_fovl = self.extract_feat(img)[0]

        fts_stride = img.size(-2) / fts_fovl.size(-2)  # fts_stride (float): 8.0/16.0/32.0;
        chip_fts_idx = chip3_fts_idx[:, idx_chip-1] / fts_stride
        feat = self.ftsfusneck([fts_fovl, fts_fovs], chip_fts_idx)

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
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
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            if 'chip4' in kwargs:  # hc-y_add0108:
                kwargs['chip4'] = kwargs['chip4'][0]
            if 'chip3_fts_idx' in kwargs:  # hc-y_add0108:
                kwargs['chip3_fts_idx'] = kwargs['chip3_fts_idx'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
