# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)
from mmcv.runner import force_fp32

from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap
from ..builder import HEADS
from .anchor_head import AnchorHead
from .yolof_head import levels_to_images, YOLOFHead

INF = 1e8


@HEADS.register_module()
class YOLOFHeadV1v1(YOLOFHead):  # hc-y_add0104:
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """
    def _init_layers(self):
        self.fus_conv = ConvModule(  # hc-y_add0104:
                    self.in_channels * 2,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg)
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 4,
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward_single(self, feature):
        feature = self.fus_conv(feature)  # hc-y_add0104:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) +
            torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg


@HEADS.register_module()
class YOLOFHeadV1v2(YOLOFHead):  # hc-y_add0106:
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """
    def _init_layers(self):
        self.fus_conv = ConvModule(  # hc-y_add0107:
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg)
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 4,
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward_single(self, feature):
        feature = self.fus_conv(feature)  # hc-y_add0104:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) +
            torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg
