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
from PIL import Image, ImageDraw, ImageFont
import math
from tools.general import xyxy2xywhn, xywh2xyxy, increment_path
from tools.plots import plot_images_v1
from mmdet.core import bbox_cxcywh_to_xyxy
# from mmdet.core.bbox.iou_calculators import fp16_clamp


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


def cluster_gt_bboxes(p_bboxes):
    """hc-y_write0326:基于 DensityPeakCluster 生成 small objects 的聚集区域;

    Args:
        p_bboxes (Tensor): gt bboxes with shape (N, 4) in (x1, y1, x2, y2) format;

    Returns:
        cluster_label (Tensor): with shape (N, 2), which cluster each gt bbox belongs.
                    一个用于指示划分到了哪个簇, 一个用于指示是否是簇中心;
    """
    num_p = len(p_bboxes)
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
        ctr_xy_dist = torch.pow(bboxes1_ctr_xy[..., :, None, :] - bboxes1_ctr_xy[..., None, :, :], 2.).sum(dim=-1)
        dist_obj = ctr_xy_dist

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
    _cluster_1st = (p_rho * p_sigma).sort(descending=True)[1][:5]  # 预设 5 个簇中心, 但最终至多保留 3 个簇中心
    cluster_1st = []
    for _idx in range(len(_cluster_1st)):
        if _idx == 0 or dist_obj[_cluster_1st[_idx], _cluster_1st[:_idx]].min() > candidate_dist_obj_thr * 3.2:
            cluster_1st.append(_cluster_1st[_idx])
    cluster_label = torch.zeros((num_p, 2), dtype=torch.int64, device=dist_obj.device)
    if len(cluster_1st) == 0:
        return cluster_label
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
            # 对于除 traffic_light 之外的其它类别, giou:*0.95, diouv:*1.05; 对于 traffic_light 类别, giou:*0.95, diouv:*0.95; 对于不区分类别, giou:*1.0, diouv:*1.0;
            if _min_dist_obj <= candidate_dist_obj_thr * 3.2:  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点是否会被划分入簇;
                cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]
            # cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]  # 不加限制地对所有点都划分入簇, tag:all
            p_rho_clustered_ind.append(_sorted_ind.item())

    return cluster_label


def inject_chips_params_to_anns(json_dir):
    """hc-y_write0421:读取 train.json, val.json, 基于 gt_bboxes 为每张图片计算 chips 参数; 并可视化画框显示效果;

    Args:
        json_dir (class::Path): ;

    Returns:
        xxx;
    """
    if "test-meta.json" in (json_dir.stem + '.json'):
        return
    last_frame_time_type = ['t-0', 't-1', 't-2', 't-3', 't-4']
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
    path_to_tmp = json_dir.parents[1] / 'imgs_vis_tmp'
    if not path_to_tmp.exists():
        path_to_tmp.mkdir(parents=True, exist_ok=True)
    # imgs_new = []
    for imgs_per_seq in imgs_per_seqs:
        imgs_per_seq.sort(key=lambda x:x['fid'])  # 依据 _img['fid'] 从小到大对 frame 排序
        for _img_idx, _img in enumerate(imgs_per_seq):
            anns_per_img = [_ann for _ann in a['annotations'] if _ann['image_id'] == _img['id']]
            if len(anns_per_img) == 0:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            _gt_anns = np.array([[_val['category_id'],] + _val['bbox'] for _val in anns_per_img], dtype=np.float64)
            _gt_anns[:, 1:3] += _gt_anns[:, 3:] / 2  # xy top-left corner to center
            _gt_anns[:, [1, 3]] /= _img['width']  # normalize x
            _gt_anns[:, [2, 4]] /= _img['height']  # normalize y
            # _ind_cls1_l = np.where(_gt_anns[:, 0] == 6)[0]  # traffic_light
            # _ind_cls2_l = np.where(_gt_anns[:, 0] != 6)[0]  # 除 traffic_light 之外的其它类别

            # if _img['sid'] != 18:
            #     break

            _chips_cf = []
            _gt_bboxes_area = _gt_anns[:, 3] * _gt_anns[:, 4]
            _sm_obj_mask = _gt_bboxes_area < 0.01  # 中小目标的面积阈值, 根据数据集及应用场景而设定, 高于该值的目标无需crop放大;
            _gt_bboxes = bbox_cxcywh_to_xyxy(torch.from_numpy(_gt_anns[_sm_obj_mask][:, 1:]))
            cluster_label = cluster_gt_bboxes(_gt_bboxes)
            path_to_img = json_dir.parents[1] / f"images_ann/{a['seq_dirs'][_img['sid']]}/{_img['name']}"
            img_src = Image.open(str(path_to_img))
            img_src_draw = DashedImageDraw(img_src)
            _cluster_color = [(0,0,255), (0,128,0), (128,0,128)]  # blue, green, purple
            for _cluster_id in [1, 2, 3]:
                _ind_cluster = torch.nonzero(cluster_label[:, 0] == _cluster_id, as_tuple=False).squeeze()
                # if _ind_cluster.numel() <= 3:  # 丢弃所包含 objects 数量少于等于 3 的簇
                if _ind_cluster.numel() < 1:
                    break
                elif _ind_cluster.numel() == 1:
                    _chip_ltrb = _gt_bboxes[_ind_cluster, :].clone()
                    _ind_cluster = [_ind_cluster, ]
                else:
                    _chip_ltrb = torch.cat((_gt_bboxes[_ind_cluster, :2].min(dim=0)[0], _gt_bboxes[_ind_cluster, 2:].max(dim=0)[0]),-1)
                _chip_ltrb[[0, 2]] *= _img['width']
                _chip_ltrb[[1, 3]] *= _img['height']
                img_src_draw.dashed_rectangle(_chip_ltrb.cpu().numpy(), dash=(8,8), outline=_cluster_color[_cluster_id-1], width=3)
                _chips_cf.append(_chip_ltrb.cpu().numpy().tolist())
                for _ind in _ind_cluster:
                    _bbox = _gt_bboxes[_ind].cpu().numpy()
                    _bbox_ctr_x = (_bbox[0] + _bbox[2])/2 * _img['width']
                    _bbox_ctr_y = (_bbox[1] + _bbox[3])/2 * _img['height']
                    img_src_draw.chord([_bbox_ctr_x-6, _bbox_ctr_y-6, _bbox_ctr_x+6, _bbox_ctr_y+6], 0, 360, fill=_cluster_color[_cluster_id-1])
                    if cluster_label[_ind, 1] == 1:
                        img_src_draw.chord([_bbox_ctr_x-3, _bbox_ctr_y-3, _bbox_ctr_x+3, _bbox_ctr_y+3], 0, 360, fill=(255,0,0))

            img_src.save(path_to_tmp / f"sid{_img['sid']}_fid{_img['fid']}_cls_euc_3cluster.jpg")
            
            for _idx_type in range(num_lft_type):
                _img_idx_nf = _img_idx + _idx_type
                _fid_nf = _img['fid'] + _idx_type*100
                if _img_idx_nf < len(imgs_per_seq) and imgs_per_seq[_img_idx_nf]['fid'] == _fid_nf:
                    if imgs_per_seq[_img_idx_nf].get('chips_lft', None) is None:
                        imgs_per_seq[_img_idx_nf]['chips_lft'] = [[] for _ in range(num_lft_type)]
                    imgs_per_seq[_img_idx_nf]['chips_lft'][_idx_type].extend(_chips_cf)
        
        # imgs_new.extend(imgs_per_seq)
    # a['images'] = imgs_new

    path_to_new_file = json_dir.parents[1] / 'annotations_focus'
    if not path_to_new_file.exists():
        path_to_new_file.mkdir(parents=True, exist_ok=True)
    json.dump(a, open(path_to_new_file  / (json_dir.stem + '_5lft_1.json'), 'w'), indent=4)


def plot_labels_on_img(json_dir):
    imgs_dir = json_dir.parents[1] / 'images'
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
            path_to_tmp = json_dir.parents[1] / f"images_ann/{a['seq_dirs'][_img['sid']]}"
            if not path_to_tmp.exists():
                path_to_tmp.mkdir(parents=True, exist_ok=True)
            plot_images_v1(None, np.concatenate((np.zeros_like(_labels[:,0:1]), _labels), -1), (_img_file, ), path_to_tmp / _img['name'], cls_names, None, 'original_image')


def main():
    dir = Path('/home/hustget/hustget_workdir/yuhangcheng/Pytorch_WorkSpace/OpenSourcePlatform/datasets')
    # annotations_dir = 'Argoverse-1.1/annotations/'
    annotations_dir = 'Argoverse-HD-mini/annotations/'
    # plot_labels_on_img(dir / 'Argoverse-HD-mini/annotations/train*.json')
    # plot_labels_on_img(dir / 'Argoverse-HD-mini/annotations_scale_type_800_sub/val*.json')
    inject_chips_params_to_anns(dir / annotations_dir / "val.json")
    print('\nfinish!')


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    main()
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD-mini
    # rm -rf *.json
