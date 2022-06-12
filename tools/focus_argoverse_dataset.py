#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file    : focus_argoverse_dataset.py
@path_to_file:
@date    : 2022/04/20 20:56
@contact :
@brief   : 为 Argoverse-HD 数据集中的每张图片生成 small objects 的聚集区域;
@intro   :
@relatedfile:
    

@annotation: hc-y_note:, hc-y_Q:, hc-y_highlight:, hc-y_add:, hc-y_modify:, c-y_write:,
"""
from unicodedata import category
import torch
import glob
import sys
from pathlib import Path

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # MMDet root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH

import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import math
from mmdet.utils import xyxy2xywh, xyxy2xywhn, xywh2xyxy, increment_path, clip_coords
from mmdet.utils import plot_images_v1
from mmdet.core import bbox_cxcywh_to_xyxy
# from mmdet.core.bbox.iou_calculators import fp16_clamp
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class DashedImageDraw(ImageDraw.ImageDraw):
    """hc-y_add0422: copied from https://stackoverflow.com/a/65893631
    PIL library (specifically PIL.ImageDraw.ImageDraw) doesn't provide 
    the functionality to draw dashed lines, so the class DashedImageDraw 
    (which extends PIL.ImageDraw.ImageDraw) is wrote, which has functionality 
    to draw dashed line and dashed rectangle.
    """
    def thick_line(self, xy, direction, fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        #direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill = fill, width = width)
        else:
            x1, y1 = xy[0]            
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1)/(dy2 - dy1)
                    a = 1/math.sqrt(1 + k**2)
                    b = (width*a - 1) /2
                else:
                    k = 0
                    b = (width - 1)/2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k*b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k*b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1)/2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1)/2)
            self.line([(x3, y3), (x4, y4)], fill = fill, width = 1)
        return   
        
    def dashed_line(self, xy, dash=(2,2), fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length**2 + y_length**2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion/length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start*x_length),
                                          round(y1 + start*y_length)),
                                         (round(x1 + end*x_length),
                                          round(y1 + end*y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2,2), outline=None, width=0):
        #xy - Sequence of [(x1, y1), (x2, y2)] where (x1, y1) is top left corner and (x2, y2) is bottom right corner
        x1, y1, x2, y2 = xy
        halfwidth1 = math.floor((width - 1)/2)
        halfwidth2 = math.ceil((width - 1)/2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1)%2        
        self.dashed_line([(x1 - halfwidth1, y1), (x2 - end_change1, y1)],
                         dash, outline, width)       
        self.dashed_line([(x2, y1 - halfwidth1), (x2, y2 - end_change1)],
                         dash, outline, width)
        self.dashed_line([(x2 + halfwidth2, y2 + odd_width_change),
                          (x1 + end_change2, y2 + odd_width_change)],
                         dash, outline, width)
        self.dashed_line([(x1 + odd_width_change, y2 + halfwidth2),
                          (x1 + odd_width_change, y1 + end_change2)],
                         dash, outline, width)
        return


def fp16_clamp(x, min=None, max=None):
    "hc-y_modify0420:copied from mmdet/core/bbox/iou_calculators/iou2d_calculator.py;"
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps_ext(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    hc-y_modify0420:modified from mmdet/core/bbox/iou_calculators/iou2d_calculator.py;
    新增了 DIoU 及其变种 DIoUv, DIoUv2 的计算;

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou', 'diou', 'diouv', 'diouv2'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou', 'diou', 'diouv', 'diouv2']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        elif mode in ['diou', 'diouv', 'diouv2']:  # hc-y_add0420:
            bboxes1_ctr_xy = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
            bboxes2_ctr_xy = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
            ctr_xy_dist = torch.pow(bboxes1_ctr_xy[..., :] - bboxes2_ctr_xy[..., :], 2.).sum(dim=-1)
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
            enclosed_diag_dist = torch.pow(enclosed_rb - enclosed_lt, 2.).sum(dim=-1)
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou', 'diou', 'diouv', 'diouv2']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])
        elif mode in ['diou', 'diouv', 'diouv2']:  # hc-y_add0420:
            bboxes1_ctr_xy = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
            bboxes2_ctr_xy = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
            ctr_xy_dist = torch.pow(bboxes1_ctr_xy[..., :, None, :] - bboxes2_ctr_xy[..., None, :, :], 2.).sum(dim=-1)
            # ctr_xy_dist_ = torch.cdist(bboxes1_ctr_xy, bboxes2_ctr_xy, p=2.)**2  # ((ctr_xy_dist_ - ctr_xy_dist) > 0.0000001).sum()
            
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])
            enclosed_diag_dist = torch.pow(enclosed_rb - enclosed_lt, 2.).sum(dim=-1)

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    elif mode in ['diou', 'diouv', 'diouv2']:  # hc-y_add0420:
        enclosed_diag_dist = torch.max(enclosed_diag_dist, eps)
        dious = ious - ctr_xy_dist / enclosed_diag_dist
        if mode == 'diou':  # the value interval of diou belongs to [-1,1]
            return dious
        elif mode == 'diouv':  # the value interval of diouv belongs to [-1,2]
            return dious + 0.8 * torch.min(area1[..., None] / area2[..., None, :], area2[..., None, :] / area1[..., None]).clamp(max=1.)
        elif mode == 'diouv2':  # the value interval of diouv2 belongs to [0,2]
            return (1. - dious) * (1 - torch.min(area1[..., None] / area2[..., None, :], area2[..., None, :] / area1[..., None]).clamp(max=1.))
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def cluster_gt_bboxes_ndarray(p_bboxes, img_wh):
    """hc-y_write0503:基于 DensityPeakCluster 生成 small objects 的聚集区域;
    注:数据在 ndarray 和 Torch.Tensor 这两种形式下的有效数字位数会不一样, e.g., p_rho 的值会有差异;

    Args:
        p_bboxes (ndarray): gt bboxes with shape (N, 4) in (x1, y1, x2, y2) format;
        img_wh (Tuple): width and height of original image;

    Returns:
        cluster_label (ndarray): with shape (N, 2), which cluster each gt bbox belongs.
                    一个用于指示划分到了哪个簇, 一个用于指示是否是簇中心;
        chips_ltrb (ndarray): with shape (<=3,4), 生成的原始 chips 参数;
        chips_ltrb_expand_new (v): with shape (<=3,4), 过滤掉被其它 chips_expand 所包含
                    的以及所包含 objects 数量少于3的 chips 后, 剩余 chips 所对应的 chips_expand 参数;
    """
    p_bboxes = p_bboxes.cpu().numpy()
    num_p = len(p_bboxes)
    if num_p <= 2:
        return np.zeros((num_p, 2), dtype=np.int64), np.empty((0, 4), dtype=p_bboxes.dtype), None
    # Step 1. compute dist_obj (distance) between all gt bboxes, and adjust the value interval from [-1,1] to [0,2] via (1. - giou);
    bboxes1_ctr_xy = (p_bboxes[..., :2] + p_bboxes[..., 2:]) / 2
    dist_obj = np.power(bboxes1_ctr_xy[..., :, None, :] - bboxes1_ctr_xy[..., None, :, :], 2.).sum(axis=-1)  # ctr_xy_dist

    # Step 2. select the \k smallest dist_obj as candidates, and compute the mean and std, set mean + std * 0.8 as the dist_obj threshold (cuttoff_distance);
    # selectable_k = min(int(num_p * 2), 36)  # 超参数, 待调节;
    selectable_k = int(num_p * 2)
    triu_inds = np.triu_indices(num_p, 1)
    candidate_dist_obj = np.sort(dist_obj[triu_inds[0], triu_inds[1]])[:selectable_k]
    candidate_dist_obj_thr = candidate_dist_obj.mean() + candidate_dist_obj.std() * 1.2  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点划分簇的先后顺序; * 1.05
    # Step 3. compute local density \rho for each gt bbox; 使用了 gaussian kernel 连续型密度;
    _p_rho = np.exp(-np.square(dist_obj / candidate_dist_obj_thr)).sum(axis=1) - 1.
    p_rho = (np.exp(-np.square(dist_obj / candidate_dist_obj_thr)) * (dist_obj <= candidate_dist_obj_thr)).sum(axis=1) - 1.  # 减去1是因为不包括与其自身之间的dist_obj
    # Step 4. compute high local density distance \sigma for each gt bbox;
    # hld_mask = np.tile(p_rho, (num_p,1)) > np.repeat(p_rho, num_p).reshape(-1, num_p)
    hld_mask = np.tile(p_rho, (num_p,1)) > np.tile(p_rho, (num_p,1)).T
    p_sigma = (hld_mask * dist_obj + ~hld_mask * dist_obj.max()).min(axis=1)
    p_sigma[p_rho.argmax()] = dist_obj.max()  # 对于最大局部密度点, 设置 \sigma 为 dist_obj 的最大值dist_obj.max()
    # Step 5. determine the cluster center according to \rho and \sigma;
    # cluster_1st = np.argsort(p_rho * p_sigma)[::-1][:3]
    _cluster_1st = np.argsort(p_rho * p_sigma)[::-1][:5]  # 依据 (p_rho * p_sigma) 从高到低, 预设 5 个簇中心, 但最终至多保留 3 个簇中心
    cluster_dist_obj_thr = candidate_dist_obj_thr * 3.2  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点是否会被划分入簇;
    cluster_1st = []
    for _idx in range(len(_cluster_1st)):  # 如果某个簇中心可以被划分到另一个簇中心 (p_rho * p_sigma) 值比其高的簇中, 则移除该簇中心;
        if _idx == 0 or dist_obj[_cluster_1st[_idx], _cluster_1st[:_idx]].min() > cluster_dist_obj_thr:
            cluster_1st.append(_cluster_1st[_idx])
    cluster_label = np.zeros((num_p, 2), dtype=np.int64)
    if len(cluster_1st) == 0:
        return cluster_label, np.empty((0, 4), dtype=p_bboxes.dtype), None
    else:
        cluster_1st = cluster_1st[:3]
    cluster_label[cluster_1st, 0] = np.arange(len(cluster_1st)) + 1
    cluster_label[cluster_1st, 1] = 1  # 簇中心
    # Step 6. cluster other gt boxes to the cluster center;
    # 方式一: 依据局部密度 $rho$ 从高到低地给各个点划分簇, 归属于距离最近的高密度点所在的簇;
    p_rho_sorted_ind = np.argsort(_p_rho)[::-1]  # 使用 _p_rho 而不是 p_rho 可以避免 p_rho 值等于零时没法排序;
    p_rho_clustered_ind = [] + cluster_1st  # sid23_fid213_cls_euc_3cluster.jpg
    for _sorted_ind in p_rho_sorted_ind:
        if cluster_label[_sorted_ind, 0] == 0:
            _argmin_dist_obj = dist_obj[_sorted_ind, p_rho_clustered_ind].argmin(axis=0)
            _min_dist_obj = dist_obj[_sorted_ind, p_rho_clustered_ind][_argmin_dist_obj]
            # 对于除 traffic_light 之外的其它类别, 略; 对于 traffic_light 类别, 略; 对于不区分类别, giou:*1.0, diouv:*1.0;
            if _min_dist_obj <= cluster_dist_obj_thr:
                cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]
            # cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]  # 不加限制地对所有点都划分入簇, tag:all
            p_rho_clustered_ind.append(_sorted_ind)

    # 计算属于同一簇的 objects 的包围框
    num_per_cluster, num_per_chip = [], []
    chips_ltrb_list = []
    for _cluster_id in range(1, len(cluster_1st)+1):
        _ind_cluster = np.where(cluster_label[:, 0] == _cluster_id)[0]
        num_per_cluster.append(len(_ind_cluster))
        if len(_ind_cluster) == 1:
            chips_ltrb_list.append(p_bboxes[_ind_cluster[0], :])
        else:
            chips_ltrb_list.append(np.concatenate((p_bboxes[_ind_cluster, :2].min(axis=0), p_bboxes[_ind_cluster, 2:].max(axis=0)),-1))
    chips_ltrb = np.stack(chips_ltrb_list, axis=0)
    chips_ltrb[:, [0, 2]] *= img_wh[0]
    chips_ltrb[:, [1, 3]] *= img_wh[1]
    chips_xywh_ = xyxy2xywh(chips_ltrb)
    chips_xywh_[:, 2:] += np.tile(chips_xywh_[:, 2:].min(axis=1, keepdims=True) * 0.4, (1,2))  # 超参数, 待调节;
    # chips_ltrb_expand = xywh2xyxy(chips_xywh_)
    # clip_coords(chips_ltrb_expand, (img_wh[1],img_wh[0]))

    if len(chips_ltrb) > 1:
        iof_chips_ = bbox_overlaps(chips_ltrb, xywh2xyxy(chips_xywh_), mode='iof', is_aligned=False)
        iof_chips_ = iof_chips_ - np.diagflat(np.diag(iof_chips_))
        chips_merge_tag = (iof_chips_.argmax(axis=1) + 1) * (iof_chips_.max(axis=1) > 0.95)  # chips_merge_id >=1 的 chip_ltrb 可以被合并;
        _chip_id_exclude, chips_ltrb_new = [], []
        for _chip_id in range(len(chips_ltrb)):  # 当前的 chip_ltrb
            if _chip_id in _chip_id_exclude:
                continue
            _chip_id_merged = np.where(chips_merge_tag == _chip_id + 1)[0]  # 被 merged 的那个 chip_ltrb
            if len(_chip_id_merged) == 0 and num_per_cluster[_chip_id] > 2:
                chips_ltrb_new.append(chips_ltrb[_chip_id])
            elif len(_chip_id_merged) == 1 and num_per_cluster[_chip_id]+num_per_cluster[_chip_id_merged[0]] > 2:
                chip_ltrb_new = np.stack((chips_ltrb[_chip_id], chips_ltrb[_chip_id_merged[0]]), axis=0)
                chip_ltrb_new = np.concatenate((chip_ltrb_new[:, :2].min(axis=0), chip_ltrb_new[:, 2:].max(axis=0)),-1)
                chips_ltrb_new.append(chip_ltrb_new)
                _chip_id_exclude.append(_chip_id_merged)
            elif len(_chip_id_merged) > 1 and num_per_cluster[_chip_id]+sum([num_per_cluster[_val] for _val in _chip_id_merged]) > 2:
                chip_ltrb_new = np.concatenate((chips_ltrb[_chip_id][None,:], chips_ltrb[_chip_id_merged]), axis=0)
                chip_ltrb_new = np.concatenate((chip_ltrb_new[:, :2].min(axis=0), chip_ltrb_new[:, 2:].max(axis=0)),-1)
                chips_ltrb_new.append(chip_ltrb_new)
                _chip_id_exclude.extend([_val for _val in _chip_id_merged])
        if len(chips_ltrb_new) > 0:
            chips_xywh_new_ = xyxy2xywh(np.stack(chips_ltrb_new, axis=0))
            chips_xywh_new_[:, 2:] += np.tile(chips_xywh_new_[:, 2:].min(axis=1, keepdims=True) * 0.4, (1,2))  # 超参数, 待调节;
        else:
            chips_xywh_new_ = None
    else:
        if num_per_cluster[0] > 2:
            chips_xywh_new_ = chips_xywh_
        else:
            chips_xywh_new_ = None
    if chips_xywh_new_ is not None:
        chips_ltrb_expand_new = xywh2xyxy(chips_xywh_new_)
        clip_coords(chips_ltrb_expand_new, (img_wh[1],img_wh[0]))
    else:
        chips_ltrb_expand_new = None

    return cluster_label, chips_ltrb, chips_ltrb_expand_new  # chips_ltrb_expand_new


def cluster_gt_bboxes(p_bboxes, img_wh):
    """hc-y_write0326:基于 DensityPeakCluster 生成 small objects 的聚集区域;

    Args:
        p_bboxes (Tensor): gt bboxes with shape (N, 4) in (x1, y1, x2, y2) format;
        img_wh (Tuple): width and height of original image;

    Returns:
        cluster_label (Tensor): with shape (N, 2), which cluster each gt bbox belongs.
                    一个用于指示划分到了哪个簇, 一个用于指示是否是簇中心;
        chips_ltrb (Tensor): with shape (<=3,4), 生成的原始 chips 参数;
        chips_ltrb_expand_new (Tensor): with shape (<=3,4), 过滤掉被其它 chips_expand 所包含
                    的以及所包含 objects 数量少于3的 chips 后, 剩余 chips 所对应的 chips_expand 参数;
    """
    num_p = len(p_bboxes)
    if num_p <= 2:
        return torch.zeros((num_p, 2), dtype=torch.int64, device=p_bboxes.device), torch.empty((0, 4), dtype=p_bboxes.dtype, device=p_bboxes.device), None
    # Step 1. compute dist_obj (distance) between all gt bboxes, and adjust the value interval from [-1,1] to [0,2] via (1. - giou);
    dist_obj_mode = 'EuclideanDist'
    if dist_obj_mode in ['giou', 'diou']:
        dist_obj = 1. - bbox_overlaps_ext(p_bboxes, p_bboxes, mode=dist_obj_mode, is_aligned=False, eps=1e-7)
    elif dist_obj_mode == 'diouv':
        dist_obj = 2. - bbox_overlaps_ext(p_bboxes, p_bboxes, mode='diouv', is_aligned=False, eps=1e-7)
    elif dist_obj_mode == 'diouv2':
        dist_obj = bbox_overlaps_ext(p_bboxes, p_bboxes, mode='diouv2', is_aligned=False, eps=1e-7)
    elif dist_obj_mode in ['EuclideanDist',]:
        bboxes1_ctr_xy = (p_bboxes[..., :2] + p_bboxes[..., 2:]) / 2
        dist_obj = torch.pow(bboxes1_ctr_xy[..., :, None, :] - bboxes1_ctr_xy[..., None, :, :], 2.).sum(dim=-1)  # ctr_xy_dist

    # Step 2. select the \k smallest dist_obj as candidates, and compute the mean and std, set mean + std * 0.8 as the dist_obj threshold (cuttoff_distance);
    selectable_k = int(num_p * 2)  # 超参数, 待调节;
    triu_inds = torch.triu_indices(num_p, num_p, 1)
    candidate_dist_obj = dist_obj[triu_inds[0], triu_inds[1]].sort(descending=False)[0][:selectable_k]
    candidate_dist_obj_thr = candidate_dist_obj.mean() + candidate_dist_obj.std() * 1.2  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点划分簇的先后顺序; * 1.05
    # Step 3. compute local density \rho for each gt bbox; 使用了 gaussian kernel 连续型密度;
    p_rho = (torch.exp(-(dist_obj / candidate_dist_obj_thr).square()) * (dist_obj <= candidate_dist_obj_thr)).sum(dim=1) - 1.  # 减去1是因为不包括与其自身之间的dist_obj
    # 删除dist_obj_triu = torch.triu(dist_obj, diagonal=1)
    # 删除rho_triu = torch.exp(-(dist_obj.triu(diagonal=1) / candidate_dist_obj_thr).square()) * torch.ones_like(dist_obj).triu(diagonal=1)
    # 删除rho = rho_triu.sum(dim=1)
    # Step 4. compute high local density distance \sigma for each gt bbox;
    hld_mask = p_rho.expand_as(dist_obj) > p_rho[:, None].expand_as(dist_obj)
    p_sigma = (hld_mask * dist_obj + ~hld_mask * dist_obj.max()).min(dim=1)[0]
    p_sigma[p_rho.argmax()] = dist_obj.max()  # 对于最大局部密度点, 设置 \sigma 为 dist_obj 的最大值dist_obj.max()
    # Step 5. determine the cluster center according to \rho and \sigma;
    # cluster_1st = (p_rho * p_sigma).sort(descending=True)[1][:3]
    _cluster_1st = (p_rho * p_sigma).sort(descending=True)[1][:5]  # 依据 (p_rho * p_sigma) 从高到低, 预设 5 个簇中心, 但最终至多保留 3 个簇中心
    cluster_dist_obj_thr = candidate_dist_obj_thr * 3.2  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点是否会被划分入簇;
    cluster_1st = []
    for _idx in range(len(_cluster_1st)):  # 如果某个簇中心可以被划分到另一个簇中心 (p_rho * p_sigma) 值比其高的簇中, 则移除该簇中心;
        if _idx == 0 or dist_obj[_cluster_1st[_idx], _cluster_1st[:_idx]].min() > cluster_dist_obj_thr:
            cluster_1st.append(_cluster_1st[_idx])
    cluster_label = torch.zeros((num_p, 2), dtype=torch.int64, device=dist_obj.device)
    if len(cluster_1st) == 0:
        return cluster_label, torch.empty((0, 4), dtype=p_bboxes.dtype, device=p_bboxes.device), None
    else:
        cluster_1st = cluster_1st[:3]
    cluster_label[cluster_1st, 0] = torch.arange(len(cluster_1st), device=dist_obj.device) + 1
    cluster_label[cluster_1st, 1] = 1  # 簇中心
    # Step 6. cluster other gt boxes to the cluster center;
    # 方式一: 依据局部密度 $rho$ 从高到低地给各个点划分簇, 归属于距离最近的高密度点所在的簇;
    p_rho_sorted_ind = p_rho.sort(descending=True)[1]
    p_rho_clustered_ind = [] + cluster_1st  # sid23_fid213_cls_euc_3cluster.jpg
    for _sorted_ind in p_rho_sorted_ind:
        if cluster_label[_sorted_ind, 0] == 0:
            _min_dist_obj, _argmin_dist_obj = dist_obj[_sorted_ind, p_rho_clustered_ind].min(dim=0)
            # 对于除 traffic_light 之外的其它类别, 略; 对于 traffic_light 类别, 略; 对于不区分类别, giou:*1.0, diouv:*1.0;
            if _min_dist_obj <= cluster_dist_obj_thr:
                cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]
            # cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]  # 不加限制地对所有点都划分入簇, tag:all
            p_rho_clustered_ind.append(_sorted_ind.item())

    # 计算属于同一簇的 objects 的包围框
    num_per_cluster, num_per_chip = [], []
    chips_ltrb_list = []
    for _cluster_id in range(1, len(cluster_1st)+1):
        _ind_cluster = torch.nonzero(cluster_label[:, 0] == _cluster_id, as_tuple=False).squeeze()
        num_per_cluster.append(_ind_cluster.numel())
        if _ind_cluster.numel() == 1:
            chips_ltrb_list.append(p_bboxes[_ind_cluster, :])
        else:
            chips_ltrb_list.append(torch.cat((p_bboxes[_ind_cluster, :2].min(dim=0)[0], p_bboxes[_ind_cluster, 2:].max(dim=0)[0]),-1))
    chips_ltrb = torch.stack(chips_ltrb_list, dim=0)
    chips_ltrb[:, [0, 2]] *= img_wh[0]
    chips_ltrb[:, [1, 3]] *= img_wh[1]
    chips_xywh_ = xyxy2xywh(chips_ltrb)
    chips_xywh_[:, 2:] += (chips_xywh_[:, 2:].min(dim=1, keepdim=True)[0] * 0.4).tile(1,2)  # 超参数, 待调节;
    # chips_ltrb_expand = xywh2xyxy(chips_xywh_)
    # clip_coords(chips_ltrb_expand, (img_wh[1],img_wh[0]))

    if len(chips_ltrb) > 1:
        iof_chips_ = bbox_overlaps_ext(chips_ltrb, xywh2xyxy(chips_xywh_), mode='iof', is_aligned=False, eps=1e-7)
        iof_chips_ = iof_chips_ - iof_chips_.diag().diag_embed()
        chips_merge_tag = (iof_chips_.max(dim=1)[1] + 1) * (iof_chips_.max(dim=1)[0] > 0.95)  # chips_merge_id >=1 的 chip_ltrb 可以被合并;
        _chip_id_exclude, chips_ltrb_new = [], []
        for _chip_id in range(len(chips_ltrb)):  # 当前的 chip_ltrb
            if _chip_id in _chip_id_exclude:
                continue
            _chip_id_merged = torch.nonzero(chips_merge_tag == _chip_id + 1, as_tuple=False).squeeze()  # 被 merged 的那个 chip_ltrb
            if _chip_id_merged.numel() == 0 and num_per_cluster[_chip_id] > 2:
                chips_ltrb_new.append(chips_ltrb[_chip_id])
            elif _chip_id_merged.numel() == 1 and num_per_cluster[_chip_id]+num_per_cluster[_chip_id_merged.item()] > 2:
                chip_ltrb_new = torch.stack((chips_ltrb[_chip_id], chips_ltrb[_chip_id_merged]), dim=0)
                chip_ltrb_new = torch.cat((chip_ltrb_new[:, :2].min(dim=0)[0], chip_ltrb_new[:, 2:].max(dim=0)[0]),-1)
                chips_ltrb_new.append(chip_ltrb_new)
                _chip_id_exclude.append(_chip_id_merged.item())
            elif _chip_id_merged.numel() > 1 and num_per_cluster[_chip_id]+sum([num_per_cluster[_val.item()] for _val in _chip_id_merged]) > 2:
                chip_ltrb_new = torch.cat((chips_ltrb[_chip_id][None,:], chips_ltrb[_chip_id_merged]), dim=0)
                chip_ltrb_new = torch.cat((chip_ltrb_new[:, :2].min(dim=0)[0], chip_ltrb_new[:, 2:].max(dim=0)[0]),-1)
                chips_ltrb_new.append(chip_ltrb_new)
                _chip_id_exclude.extend([_val.item() for _val in _chip_id_merged])
        if len(chips_ltrb_new) > 0:
            chips_xywh_new_ = xyxy2xywh(torch.stack(chips_ltrb_new, dim=0))
            chips_xywh_new_[:, 2:] += (chips_xywh_new_[:, 2:].min(dim=1, keepdim=True)[0] * 0.4).tile(1,2)  # 超参数, 待调节;
        else:
            chips_xywh_new_ = None
    else:
        if num_per_cluster[0] > 2:
            chips_xywh_new_ = chips_xywh_
        else:
            chips_xywh_new_ = None
    if chips_xywh_new_ is not None:
        chips_ltrb_expand_new = xywh2xyxy(chips_xywh_new_)
        clip_coords(chips_ltrb_expand_new, (img_wh[1],img_wh[0]))
    else:
        chips_ltrb_expand_new = None

    return cluster_label, chips_ltrb, chips_ltrb_expand_new


def inject_chips_params_to_anns(json_dir, flag_img_vis=False):
    """hc-y_write0421:读取 annotations_focus/train.json, 基于 gt_bboxes 为每张图片计算 chips 参数; 并可视化画框显示效果;

    Args:
        json_dir (class::Path): ;

    Returns:
        xxx;
    """
    if "test-meta.json" in (json_dir.stem + '.json'):
        return
    last_frame_time_type = ['t-0', 't-1', 't-2', 't-3', 't-4']
    # lf: last frame; cf: current frame; nf: next frame;
    num_lft_type = len(last_frame_time_type)
    with open(json_dir, 'r') as f:
        a = json.load(f)
    
    # 区分开 frame 隶属于不同的 sequences
    imgs_per_seqs = [[] for _ in range(len(a['seq_dirs']))]
    sid_list = []
    for _img in a['images']:
        if _img['sid'] not in sid_list:
            sid_list.append(_img['sid'])
        imgs_per_seqs[sid_list.index(_img['sid'])].append(_img)

    cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
    if flag_img_vis:
        path_to_tmp = json_dir.parents[1] / f"imgs_vis_{json_dir.stem}_euc_1.2_3.2_3"
        if not path_to_tmp.exists():
            path_to_tmp.mkdir(parents=True, exist_ok=True)
    # imgs_new = []
    for imgs_per_seq in imgs_per_seqs:
        imgs_per_seq.sort(key=lambda x:x['fid'])  # 依据 _img['fid'] 从小到大对 frame 排序
        for _img_idx, _img in enumerate(tqdm(imgs_per_seq, desc='Processing %s' % imgs_per_seq[0]['sid'])):
            anns_per_img = [_ann for _ann in a['annotations'] if _ann['image_id'] == _img['id']]
            if len(anns_per_img) == 0:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            _gt_anns = np.array([[_val['category_id'],] + _val['bbox'] for _val in anns_per_img], dtype=np.float64)
            _gt_anns[:, 1:3] += _gt_anns[:, 3:] / 2  # xy top-left corner to center
            _gt_anns_ = np.copy(_gt_anns)
            _gt_anns[:, [1, 3]] /= _img['width']  # normalize x
            _gt_anns[:, [2, 4]] /= _img['height']  # normalize y
            # _ind_cls1_l = np.where(_gt_anns[:, 0] == 6)[0]  # traffic_light
            # _ind_cls2_l = np.where(_gt_anns[:, 0] != 6)[0]  # 除 traffic_light 之外的其它类别

            # if _img['sid'] != 0 or _img['fid'] != 100:
            #     continue

            _chips_cf = []
            _gt_bboxes_area = _gt_anns[:, 3] * _gt_anns[:, 4]
            _sm_obj_mask = _gt_bboxes_area < 0.01  # 中小目标的面积阈值, 根据数据集及应用场景而设定, 高于该值的目标无需crop放大;
            _gt_bboxes_sm = bbox_cxcywh_to_xyxy(torch.from_numpy(_gt_anns[_sm_obj_mask][:, 1:]))
            img_wh = (_img['width'], _img['height'])
            cluster_label_ndarray, chips_ltrb_ndarray, chips_ltrb_expand_new_ndarray = cluster_gt_bboxes_ndarray(_gt_bboxes_sm, img_wh)
            cluster_label, chips_ltrb = torch.from_numpy(cluster_label_ndarray), torch.from_numpy(chips_ltrb_ndarray)
            chips_ltrb_expand_new = torch.from_numpy(chips_ltrb_expand_new_ndarray) if chips_ltrb_expand_new_ndarray is not None else chips_ltrb_expand_new_ndarray
            # cluster_label_, chips_ltrb_, chips_ltrb_expand_new_ = cluster_gt_bboxes(_gt_bboxes_sm, img_wh)
            # if len(chips_ltrb) != len(chips_ltrb_) or not torch.all(chips_ltrb == chips_ltrb_):
                # print(f"sid{_img['sid']}_fid{_img['fid']}_cls_euc_3cluster.jpg")
            
            if flag_img_vis:
                path_to_img = json_dir.parents[1] / f"images_ann/{a['seq_dirs'][_img['sid']]}/{_img['name']}"
                img_src = Image.open(str(path_to_img))
                img_src_draw = DashedImageDraw(img_src)
                _cluster_color = [(0,0,255), (0,128,0), (128,0,128), (0,0,0), (255,255,255)]  # blue, green, purple, black, white
                for i, _bbox in enumerate(_gt_anns_[_sm_obj_mask][:, 1:]):
                    if cluster_label[i, 0] > 0:
                        img_src_draw.chord([_bbox[0]-6, _bbox[1]-6, _bbox[0]+6, _bbox[1]+6], 0, 360, fill=_cluster_color[cluster_label[i, 0]-1])
                    if cluster_label[i, 1] == 1:
                        img_src_draw.chord([_bbox[0]-3, _bbox[1]-3, _bbox[0]+3, _bbox[1]+3], 0, 360, fill=(255,0,0))
                for i, _chip_ltrb in enumerate(chips_ltrb.cpu().numpy()):
                    img_src_draw.dashed_rectangle(_chip_ltrb, dash=(8,8), outline=_cluster_color[i], width=3)
                if chips_ltrb_expand_new is not None:
                    _chips_cf.extend(chips_ltrb_expand_new.cpu().numpy().tolist())
                    for _chip_ltrb_expand_new in chips_ltrb_expand_new.cpu().numpy():
                        img_src_draw.dashed_rectangle(_chip_ltrb_expand_new, dash=(8,8), outline=_cluster_color[3], width=3)
                img_src.save(path_to_tmp / f"sid{_img['sid']}_fid{_img['fid']}_cls_euc_3cluster.jpg")
            else:
                if chips_ltrb_expand_new is not None:
                    _chips_cf.extend(chips_ltrb_expand_new.cpu().numpy().tolist())
            
            for _idx_type in range(num_lft_type):
                _img_idx_nf = _img_idx + _idx_type
                _fid_nf = _img['fid'] + _idx_type*1  # hc-y_TODO: *100 改为 *1
                if _img_idx_nf < len(imgs_per_seq) and imgs_per_seq[_img_idx_nf]['fid'] == _fid_nf:
                    if imgs_per_seq[_img_idx_nf].get('chips_lft', None) is None:
                        imgs_per_seq[_img_idx_nf]['chips_lft'] = [[] for _ in range(num_lft_type)]
                    imgs_per_seq[_img_idx_nf]['chips_lft'][_idx_type].extend(_chips_cf)
        
        # imgs_new.extend(imgs_per_seq)
    # a['images'] = imgs_new

    path_to_new_file = json_dir.parents[1] / 'annotations_focus'
    if not path_to_new_file.exists():
        path_to_new_file.mkdir(parents=True, exist_ok=True)
    json.dump(a, open(path_to_new_file  / (json_dir.stem + '_5lft_euc_1.2_3.2.json'), 'w'), indent=4)


def generate_chips_dataset(json_dir):
    """hc-y_write0430:读取 annotations_focus/train.json, 依据 chips 参数从原图中 crop 出 chips 生成 chips_dataset;

    Args:
        json_dir (class::Path): ;

    Returns:
        xxx;
    """
    with open(json_dir, 'r') as f:
        a = json.load(f)

    import cv2
    imgs_new, anns_new = [], []
    img_counts, anns_count = 0, 0
    for _img in tqdm(a['images'], desc='Chips for each Image'):
        anns_per_img = [_ann for _ann in a['annotations'] if _ann['image_id'] == _img['id']]
        if len(anns_per_img) == 0:
            continue
        # The COCO box format is [top left x, top left y, width, height]
        _gt_bboxes = np.array([_val['bbox'] for _val in anns_per_img], dtype=np.float64)
        _gt_bboxes[:, 2:] += _gt_bboxes[:, :2]  # width,height to bottom right x,y
        path_to_img = json_dir.parents[1] / f"images/{a['seq_dirs'][_img['sid']]}/{_img['name']}"
        path_to_chip = json_dir.parents[1] / f"images_chip/{a['seq_dirs'][_img['sid']]}"
        if not path_to_chip.exists():
            path_to_chip.mkdir(parents=True, exist_ok=True)
        img_src = cv2.imread(str(path_to_img))  # HWC BGR
        # lf: last frame; cf: current frame; nf: next frame;
        # hc-y_TODO: train时, 可以考虑给 chips_lf 施加一个随机抖动;
        chips_lf = _img['chips_lft'][0]
        for i in range(len(chips_lf)):
            x1a0,y1a0,x2a0,y2a0 = (int(_val) for _val in chips_lf[i])
            chip_img = img_src[y1a0:y2a0, x1a0:x2a0]
            gt_bboxes_clipped = _gt_bboxes.copy()
            np.clip(gt_bboxes_clipped[:,0::2], x1a0, x2a0, out=gt_bboxes_clipped[:,0::2])
            np.clip(gt_bboxes_clipped[:,1::2], y1a0, y2a0, out=gt_bboxes_clipped[:,1::2])
            ious_itself = bbox_overlaps(gt_bboxes_clipped, _gt_bboxes, mode='iou', is_aligned=True)
            idx_rest = np.where(ious_itself >= 0.5)[0]  # 当gt bbox被clip掉大部分比例时, 直接删除该gt bbox;
            gt_bboxes_clipped_area = gt_bboxes_clipped[:, 2]*gt_bboxes_clipped[:, 3]
            gt_bboxes_clipped[:, 2:] -= gt_bboxes_clipped[:, :2]
            gt_bboxes_clipped[:, 0] -= x1a0
            gt_bboxes_clipped[:, 1] -= y1a0
            for j in idx_rest:
                ann_new = anns_per_img[j].copy()
                ann_new['id'] = anns_count
                ann_new['image_id'] = img_counts
                ann_new['bbox'] = gt_bboxes_clipped[j].tolist()
                ann_new['area'] = gt_bboxes_clipped_area[j]
                anns_new.append(ann_new)
                anns_count += 1
            img_new = _img.copy()
            img_new['id'] = img_counts
            _img_name = img_new['name'].rsplit('.', 1)
            img_new['name'] = _img_name[0] + f"_{i}." + _img_name[1]
            cv2.imwrite(str(path_to_chip / f"{img_new['name']}"), chip_img)
            img_new['width'] = x2a0 - x1a0
            img_new['height'] = y2a0 - y1a0
            img_new.pop('chips_lft')
            imgs_new.append(img_new)
            img_counts += 1
    
    a['images'] = imgs_new
    a['annotations'] = anns_new
    path_to_new_file = json_dir.parents[1] / 'annotations_chip'
    if not path_to_new_file.exists():
        path_to_new_file.mkdir(parents=True, exist_ok=True)
    json.dump(a, open(path_to_new_file  / (json_dir.stem + '_chip.json'), 'w'), indent=4)


def plot_labels_on_img(json_dir, imgs_folder='images'):
    imgs_dir = json_dir.parents[1] / imgs_folder
    json_files = sorted(glob.glob(str(json_dir), recursive=False))
    for _json_file in json_files:
        with open(_json_file, 'r') as f:
            a = json.load(f)
        cls_names = dict()
        for _val in a['categories']:
            cls_names[_val['id']]=_val['name']
        for _img in a['images']:
            anns_per_img = [_ann for _ann in a['annotations'] if _ann['image_id'] == _img['id']]
            if len(anns_per_img) == 0:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            _labels = np.array([[_val['category_id'],] + _val['bbox'] for _val in anns_per_img], dtype=np.float64)
            _labels[:, 1:3] += _labels[:, 3:5] / 2  # xy top-left corner to center
            _labels[:, [1, 3]] /= _img['width']  # normalize x
            _labels[:, [2, 4]] /= _img['height']  # normalize y
            _img_file = str(imgs_dir) + '/' + a['seq_dirs'][_img['sid']] + '/' + _img['name']
            path_to_tmp = json_dir.parents[1] / f"{imgs_folder}_ann/{a['seq_dirs'][_img['sid']]}"
            if not path_to_tmp.exists():
                path_to_tmp.mkdir(parents=True, exist_ok=True)
            plot_images_v1(None, np.concatenate((np.zeros_like(_labels[:,0:1]), _labels), -1), (_img_file, ), path_to_tmp / _img['name'], cls_names, None, 'original_image')


def main():
    # dir = Path('/home/hustget/hustget_workdir/yuhangcheng/Pytorch_WorkSpace/OpenSourcePlatform/datasets')
    dir = Path('/media/hustget/HUSTGET/amax/HUSTGET_users/yuhangcheng/OpenSourcePlatform/datasets')
    annotations_dir = 'Argoverse-1.1/annotations'
    # annotations_dir = 'Argoverse-HD-mini/annotations'
    str_train = 'train'
    # plot_labels_on_img(dir / annotations_dir / f"{str_train}*.json", imgs_folder='images')
    inject_chips_params_to_anns(dir / annotations_dir / f"{str_train}.json", flag_img_vis=False)  # 耗时约46min
    # annotations_dir_f = annotations_dir + '_focus'
    # generate_chips_dataset(dir / annotations_dir_f / f"{str_train}_5lft_euc_1.2_3.2.json")  # 耗时约64min
    # annotations_dir_c = annotations_dir + '_chip'
    # plot_labels_on_img(dir / annotations_dir_c / f"{str_train}*.json", imgs_folder='images_chip')
    print('\nfinish!')


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    main()
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD-mini
    # rm -rf *.json
