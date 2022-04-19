# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)
from torch.nn import BatchNorm2d

from ..builder import NECKS


@NECKS.register_module()
class FtsFusNeckV1v1(nn.Module):  # hc-y_add0107:
    """FtsFus Neck for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
    """

    def __init__(self, in_channels, out_channels, ftsfus_type, 
        norm_cfg=dict(type='BN', requires_grad=True)):
        super(FtsFusNeckV1v1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ftsfus_type = ftsfus_type
        self.norm_cfg = norm_cfg
        if self.ftsfus_type in ['FtsFusV0v1', 'FtsFusV1v1', 'FtsFusV2v1', 'FtsFusV3v1']:
            self._init_layers()

    def _init_layers(self):
        if self.ftsfus_type in ['FtsFusV0v1', 'FtsFusV1v1']:
            self.fus_conv = ConvModule(  # hc-y_add0107:
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg)
        elif self.ftsfus_type in ['FtsFusV2v1', 'FtsFusV3v1']:
            self.fus_conv = ConvModule(  # hc-y_add0107:
                self.in_channels * 2,
                self.out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.fus_conv.modules():
            if isinstance(m, nn.Conv2d):
                # normal_init(m, mean=0, std=0.01)
                caffe2_xavier_init(m)
            if is_norm(m):
                constant_init(m, 1)

    def forward(self, feature, chip_fts_idx=None):
        if self.ftsfus_type == 'FtsFusV0v1':  # hc-y_add0109:
            fts_fused = self.fus_conv(feature[0])
        elif self.ftsfus_type == 'FtsFusV1v1':
            fts_fused = self._ftsfut_fovs_v1v1_singlelayer(feature, chip_fts_idx)
            fts_fused = self.fus_conv(fts_fused)
        elif self.ftsfus_type == 'FtsFusV2v1':
            fts_fused = self._ftsfut_fovs_v2v1_singlelayer(feature, chip_fts_idx)
        elif self.ftsfus_type == 'FtsFusV3v1':
            fts_fused = self._ftsfut_fovs_v3v1_singlelayer(feature, chip_fts_idx)
            fts_fused = self.fus_conv(fts_fused)
        return fts_fused,

    def _ftsfut_fovs_v1v1_singlelayer(self, ftsfus_in, chip_fts_idx):
        """
        hc-y_add0107:对chip with different FoVs经由 backbone, neck 提取得到的特征执行特征融合; 该函数只处理单个layer;
        Arguments:
            ftsfus_in (List(Tensor)): [fts_fovl, fts_fovs], each Tensor with shape torch.Size([bs, c, h, w]);
            chip_fts_idx (Tensor): outer tuple 对应于 each img; for each chip, 由(x1aa, y1aa, waa, haa), (0, 0, wbb, hbb)构成;
        Returns:
            ftsfus_out (Tensor): 融合之后的特征;
        """
        fts_fovl, fts_fovs = ftsfus_in
        # from tools.general import increment_path, mmdet_imdenormalize
        # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
        # import cv2
        ftsfus_out_list = []
        for idx_img in range(len(chip_fts_idx)):
            _hw_fts_fovl = fts_fovl[idx_img].shape[-2:]
            _olp_x1y1wh = chip_fts_idx[idx_img].round().to(torch.long)
            _fts_fovl_inter = fts_fovl[idx_img:(idx_img+1), :, _olp_x1y1wh[0,1]:(_olp_x1y1wh[0,1]+_olp_x1y1wh[0,3]), _olp_x1y1wh[0,0]:(_olp_x1y1wh[0,0]+_olp_x1y1wh[0,2])]
            # _fts_fovs_inter = fts_fovs[idx_img:(idx_img+1), :, _olp_x1y1wh[1,1]:(_olp_x1y1wh[1,1]+_olp_x1y1wh[1,3]), _olp_x1y1wh[1,0]:(_olp_x1y1wh[1,0]+_olp_x1y1wh[1,2])]

            _fts_fovs = fts_fovs[idx_img:(idx_img+1)]
            _dst_hw = _fts_fovl_inter.shape[-2:]
            _src_hw, _src_patch_x1y1wh = _fts_fovs.shape[-2:], chip_fts_idx[idx_img][1]
            _yy = (torch.arange(0, _dst_hw[0], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[3] / _dst_hw[0] + _src_patch_x1y1wh[1]
            _xx = (torch.arange(0, _dst_hw[1], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[2] / _dst_hw[1] + _src_patch_x1y1wh[0]
            _grid_y = (_yy / _src_hw[0] * 2 - 1)[None, :, None].expand(1, _dst_hw[0], _dst_hw[1])
            _grid_x = (_xx / _src_hw[1] * 2 - 1)[None, None, :].expand(1, _dst_hw[0], _dst_hw[1])
            _fts_fovs_inter_resized = F.grid_sample(_fts_fovs, torch.stack([_grid_x, _grid_y], dim=3), align_corners=False)
            
            _fts_fus_inter = torch.stack((_fts_fovl_inter, _fts_fovs_inter_resized),0).mean(dim=0)
            _m_zeropad = nn.ZeroPad2d((_olp_x1y1wh[0,0], _hw_fts_fovl[1]-_olp_x1y1wh[0,2]-_olp_x1y1wh[0,0], _olp_x1y1wh[0,1], _hw_fts_fovl[0]-_olp_x1y1wh[0,3]-_olp_x1y1wh[0,1]))
            ftsfus_out_list.append((1-_m_zeropad(torch.ones_like(_fts_fus_inter))) * fts_fovl[idx_img:(idx_img+1)] + _m_zeropad(_fts_fus_inter))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl.jpg'), mmdet_imdenormalize(fts_fovl[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl_inter.jpg'), mmdet_imdenormalize(_fts_fovl_inter[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs.jpg'), mmdet_imdenormalize(fts_fovs[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_inter_resized.jpg'), mmdet_imdenormalize(_fts_fovs_inter_resized[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_inter_fused.jpg'), mmdet_imdenormalize(_fts_fus_inter[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fus_zeropad.jpg'), mmdet_imdenormalize(_m_zeropad(_fts_fus_inter)[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl_outer.jpg'), mmdet_imdenormalize(((1-_m_zeropad(torch.ones_like(_fts_fus_inter))) * fts_fovl[idx_img:(idx_img+1)])[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fus.jpg'), mmdet_imdenormalize(ftsfus_out_list[0][0].cpu().numpy().transpose((1, 2, 0))))
        return torch.cat(ftsfus_out_list, 0)

    def _ftsfut_fovs_v2v1_singlelayer(self, ftsfus_in, chip_fts_idx):
        """
        hc-y_add0107:对chip with different FoVs经由 backbone, neck 提取得到的特征执行特征融合; 该函数只处理单个layer;
        Arguments:
            ftsfus_in (List(Tensor)): [fts_fovl, fts_fovs], each Tensor with shape torch.Size([bs, c, h, w]);
            chip_fts_idx (Tensor): outer tuple 对应于 each img; for each chip, 由(x1aa, y1aa, waa, haa), (0, 0, wbb, hbb)构成;
        Returns:
            ftsfus_out (Tensor): 融合之后的特征;
        """
        fts_fovl, fts_fovs = ftsfus_in
        # from tools.general import increment_path, mmdet_imdenormalize
        # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
        # import cv2
        ftsfus_out_list = []
        for idx_img in range(len(chip_fts_idx)):
            _hw_fts_fovl = fts_fovl[idx_img].shape[-2:]
            _olp_x1y1wh = chip_fts_idx[idx_img].round().to(torch.long)
            _fts_fovl_inter = fts_fovl[idx_img:(idx_img+1), :, _olp_x1y1wh[0,1]:(_olp_x1y1wh[0,1]+_olp_x1y1wh[0,3]), _olp_x1y1wh[0,0]:(_olp_x1y1wh[0,0]+_olp_x1y1wh[0,2])]
            # _fts_fovs_inter = fts_fovs[idx_img:(idx_img+1), :, _olp_x1y1wh[1,1]:(_olp_x1y1wh[1,1]+_olp_x1y1wh[1,3]), _olp_x1y1wh[1,0]:(_olp_x1y1wh[1,0]+_olp_x1y1wh[1,2])]

            _fts_fovs = fts_fovs[idx_img:(idx_img+1)]
            _dst_hw = _fts_fovl_inter.shape[-2:]
            _src_hw, _src_patch_x1y1wh = _fts_fovs.shape[-2:], chip_fts_idx[idx_img][1]
            _yy = (torch.arange(0, _dst_hw[0], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[3] / _dst_hw[0] + _src_patch_x1y1wh[1]
            _xx = (torch.arange(0, _dst_hw[1], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[2] / _dst_hw[1] + _src_patch_x1y1wh[0]
            _grid_y = (_yy / _src_hw[0] * 2 - 1)[None, :, None].expand(1, _dst_hw[0], _dst_hw[1])
            _grid_x = (_xx / _src_hw[1] * 2 - 1)[None, None, :].expand(1, _dst_hw[0], _dst_hw[1])
            _fts_fovs_inter_resized = F.grid_sample(_fts_fovs, torch.stack([_grid_x, _grid_y], dim=3), align_corners=False)
            
            _fts_fus_inter = self.fus_conv(torch.cat((_fts_fovl_inter, _fts_fovs_inter_resized),1))
            _m_zeropad = nn.ZeroPad2d((_olp_x1y1wh[0,0], _hw_fts_fovl[1]-_olp_x1y1wh[0,2]-_olp_x1y1wh[0,0], _olp_x1y1wh[0,1], _hw_fts_fovl[0]-_olp_x1y1wh[0,3]-_olp_x1y1wh[0,1]))
            ftsfus_out_list.append((1-_m_zeropad(torch.ones_like(_fts_fus_inter))) * fts_fovl[idx_img:(idx_img+1)] + _m_zeropad(_fts_fus_inter))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl.jpg'), mmdet_imdenormalize(fts_fovl[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl_inter.jpg'), mmdet_imdenormalize(_fts_fovl_inter[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs.jpg'), mmdet_imdenormalize(fts_fovs[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_inter_resized.jpg'), mmdet_imdenormalize(_fts_fovs_inter_resized[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_inter_fused.jpg'), mmdet_imdenormalize(_fts_fus_inter[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fus_zeropad.jpg'), mmdet_imdenormalize(_m_zeropad(_fts_fus_inter)[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl_outer.jpg'), mmdet_imdenormalize((1-_m_zeropad(torch.ones_like(_fts_fus_inter))) * fts_fovl[idx_img:(idx_img+1)][0].cpu().numpy().transpose((1, 2, 0))))
        return torch.cat(ftsfus_out_list, 0)

    def _ftsfut_fovs_v3v1_singlelayer(self, ftsfus_in, chip_fts_idx):
        """
        hc-y_add0107:对chip with different FoVs经由 backbone, neck 提取得到的特征执行特征融合; 该函数只处理单个layer;
        Arguments:
            ftsfus_in (List(Tensor)): [fts_fovl, fts_fovs], each Tensor with shape torch.Size([bs, c, h, w]);
            chip_fts_idx (Tensor): outer tuple 对应于 each img; for each chip, 由(x1aa, y1aa, waa, haa), (0, 0, wbb, hbb)构成;
        Returns:
            ftsfus_out (Tensor): 融合之后的特征;
        """
        fts_fovl, fts_fovs = ftsfus_in
        # from tools.general import increment_path, mmdet_imdenormalize
        # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
        # import cv2
        fts_fovs_zeropad_list = []
        for idx_img in range(len(chip_fts_idx)):
            _hw_fts_fovl = fts_fovl[idx_img].shape[-2:]
            _olp_x1y1wh = chip_fts_idx[idx_img].round().to(torch.long)
            _fts_fovl_inter = fts_fovl[idx_img:(idx_img+1), :, _olp_x1y1wh[0,1]:(_olp_x1y1wh[0,1]+_olp_x1y1wh[0,3]), _olp_x1y1wh[0,0]:(_olp_x1y1wh[0,0]+_olp_x1y1wh[0,2])]
            # _fts_fovs_inter = fts_fovs[idx_img:(idx_img+1), :, _olp_x1y1wh[1,1]:(_olp_x1y1wh[1,1]+_olp_x1y1wh[1,3]), _olp_x1y1wh[1,0]:(_olp_x1y1wh[1,0]+_olp_x1y1wh[1,2])]

            _fts_fovs = fts_fovs[idx_img:(idx_img+1)]
            _dst_hw = _fts_fovl_inter.shape[-2:]
            _src_hw, _src_patch_x1y1wh = _fts_fovs.shape[-2:], chip_fts_idx[idx_img][1]
            _yy = (torch.arange(0, _dst_hw[0], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[3] / _dst_hw[0] + _src_patch_x1y1wh[1]
            _xx = (torch.arange(0, _dst_hw[1], device=fts_fovl.device).to(fts_fovl.dtype) + 0.5) * _src_patch_x1y1wh[2] / _dst_hw[1] + _src_patch_x1y1wh[0]
            _grid_y = (_yy / _src_hw[0] * 2 - 1)[None, :, None].expand(1, _dst_hw[0], _dst_hw[1])
            _grid_x = (_xx / _src_hw[1] * 2 - 1)[None, None, :].expand(1, _dst_hw[0], _dst_hw[1])
            _fts_fovs_inter_resized = F.grid_sample(_fts_fovs, torch.stack([_grid_x, _grid_y], dim=3), align_corners=False)
            
            _m_zeropad = nn.ZeroPad2d((_olp_x1y1wh[0,0], _hw_fts_fovl[1]-_olp_x1y1wh[0,2]-_olp_x1y1wh[0,0], _olp_x1y1wh[0,1], _hw_fts_fovl[0]-_olp_x1y1wh[0,3]-_olp_x1y1wh[0,1]))
            fts_fovs_zeropad_list.append(_m_zeropad(_fts_fovs_inter_resized))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl.jpg'), mmdet_imdenormalize(fts_fovl[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovl_inter.jpg'), mmdet_imdenormalize(_fts_fovl_inter[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs.jpg'), mmdet_imdenormalize(fts_fovs[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_inter_resized.jpg'), mmdet_imdenormalize(_fts_fovs_inter_resized[0].cpu().numpy().transpose((1, 2, 0))))
            # cv2.imwrite(str(path_to_tmp / f'img0_fovs_zeropad.jpg'), mmdet_imdenormalize(_m_zeropad(_fts_fovs_inter_resized)[0].cpu().numpy().transpose((1, 2, 0))))
        fts_fovs_zeropad = torch.cat(fts_fovs_zeropad_list, 0)
        return torch.cat((fts_fovl, fts_fovs_zeropad),1)
