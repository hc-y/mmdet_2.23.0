#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file    : densitypeakcluster.py
@path_to_file:
@date    : 2022/03/26
@contact :
@brief   : 基于 DensityPeakCluster 密度最大值聚类算法,为 Argoverse-HD 数据集中的每张图片生成 small objects 的聚集区域;
@intro   :
@relatedfile:
    

@annotation: hc-y_note:, hc-y_Q:, hc-y_highlight:, hc-y_add:, hc-y_modify:, c-y_write:,
"""
import os
import numpy as np
import torch
from pathlib import Path
# from mmdet.core import bbox_cxcywh_to_xyxy, bbox_overlaps


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

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

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

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
    dist_obj_mode = 'giou'
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
    flag_cluster = 'flag_cluster_v1'
    if flag_cluster == 'flag_cluster_v1':
        p_rho_sorted_ind = p_rho.sort(descending=True)[1]
        p_rho_clustered_ind = [] + cluster_1st
        for _sorted_ind in p_rho_sorted_ind:
            if cluster_label[_sorted_ind, 0] == 0:
                _min_dist_obj, _argmin_dist_obj = dist_obj[_sorted_ind, p_rho_clustered_ind].min(dim=0)
                # 对于除 traffic_light 之外的其它类别, 略; 对于 traffic_light 类别, 略; 对于不区分类别, giou:*1.0, diouv:*1.0;
                if _min_dist_obj <= candidate_dist_obj_thr * 3.2:  # 超参数, 待调节; 该参数会影响到 Step 6 中各个点是否会被划分入簇;
                    cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]
                # cluster_label[_sorted_ind, 0] = cluster_label[p_rho_clustered_ind[_argmin_dist_obj], 0]  # 不加限制地对所有点都划分入簇, tag:all
                p_rho_clustered_ind.append(_sorted_ind.item())
    elif flag_cluster == 'flag_cluster_v2':
        # 方式二: 由簇中心向四周发散, 依据离簇中心的距离从近到远地给各个点划分簇;
        _ind_rest_1 = torch.nonzero(cluster_label[:, 0] == 0, as_tuple=False).squeeze()
        if _ind_rest_1.numel() > 0:
            _min_dist_obj, _argmin_dist_obj = dist_obj[_ind_rest_1][:, cluster_1st].min(dim=1)
            _cluster_2nd_mask = _min_dist_obj < candidate_dist_obj_thr * 1.0  # 超参数, 待调节;
            cluster_label[_ind_rest_1[_cluster_2nd_mask], 0] = cluster_label[cluster_1st, 0][_argmin_dist_obj[_cluster_2nd_mask]]
            cluster_label[_ind_rest_1[_cluster_2nd_mask], 1] = 2  # 归属于距离最近的cluster_1st点所在的簇
            cluster_2nd = _ind_rest_1[_cluster_2nd_mask]
            _ind_rest_2 = _ind_rest_1[~_cluster_2nd_mask]

            if cluster_2nd.numel() > 0 and _ind_rest_2.numel() > 0:
                _min_dist_obj, _argmin_dist_obj = dist_obj[_ind_rest_2][:, cluster_2nd].min(dim=1)
                _cluster_3rd_mask = _min_dist_obj < candidate_dist_obj_thr * 1.0  # 超参数, 待调节;
                cluster_label[_ind_rest_2[_cluster_3rd_mask], 0] = cluster_label[cluster_2nd, 0][_argmin_dist_obj[_cluster_3rd_mask]]
                cluster_label[_ind_rest_2[_cluster_3rd_mask], 1] = 3  # 归属于距离最近的cluster_2nd点所在的簇
                cluster_3rd = _ind_rest_2[_cluster_3rd_mask]
                _ind_rest_3 = _ind_rest_2[~_cluster_3rd_mask]

                if cluster_3rd.numel() > 0 and _ind_rest_3.numel() > 0:
                    _min_dist_obj, _argmin_dist_obj = dist_obj[_ind_rest_3][:, cluster_3rd].min(dim=1)
                    _cluster_4th_mask = _min_dist_obj < candidate_dist_obj_thr * 1.0  # 超参数, 待调节;
                    cluster_label[_ind_rest_3[_cluster_4th_mask], 0] = cluster_label[cluster_3rd, 0][_argmin_dist_obj[_cluster_4th_mask]]
                    cluster_label[_ind_rest_3[_cluster_4th_mask], 1] = 4  # 归属于距离最近的cluster_3rd点所在的簇
                    # cluster_4th = _ind_rest_3[_cluster_4th_mask]
                    # _ind_rest_4 = _ind_rest_3[~_cluster_4th_mask]
    return cluster_label

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    return [x.replace('.jpg', '.txt') for x in img_paths]  # hc-y_note1118:sb.join(['1','2', '3'])

def main():
    img_paths = ['./my_workspace/demo-test/ring_front_center_315984811296183496.jpg', ]
    label_files = img2label_paths(img_paths)

    # 读取 img_label 内容
    for lb_file in label_files:
        with open(lb_file, 'r') as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l = np.array(l, dtype=np.float32)
        nl = len(l)
        if nl:
            assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
            assert (l >= 0).all(), f'negative label values {l[l < 0]}'
            assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
            l = np.unique(l, axis=0)  # remove duplicate rows
        else:  # label empty
            l = np.zeros((0, 5), dtype=np.float32)
    

    cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
    _ind_cls1_l = np.where(l[:, 0] == 6)[0]  # traffic_light
    _ind_cls2_l = np.where(l[:, 0] != 6)[0]  # 除 traffic_light 之外的其它类别

    path_to_tmp = Path('./my_workspace/demo-test/')
    # from tools.plots import plot_images_v1
    # plot_images_v1(None, np.concatenate((np.zeros_like(l[l_ind_cls_2][:,0:1]), l[l_ind_cls_2]), -1), (str(path_to_tmp / f'ring_front_center_315984811296183496.jpg'), ), path_to_tmp / f'img_cls2.jpg', cls_names, None, 'original_image')
    # import cv2
    # img_cls2 = cv2.imread(str(path_to_tmp / f'img_cls2.jpg'))[:,:,::-1]  # HWC BGR  --> HWC RGB
    from PIL import Image, ImageDraw, ImageFont
    # img_cls2 = Image.fromarray(img_cls2)
    # img_cls2 = Image.open(str(path_to_tmp / f'img_cls2.jpg'))
    img_cls2 = Image.open(str(path_to_tmp / f'val_batch19_labels.jpg'))
    img_cls_2_draw = ImageDraw.Draw(img_cls2)
    img_wh = img_cls2.size

    if False:
        box = [1920*0.2, 1200*0.2, 1920*0.8, 1200*0.8]
        # txt_font = ImageFont.truetype('C:\\Users\\YHC\\AppData\\Roaming\\Ultralytics\\Arial.ttf', 20)
        txt_font = ImageFont.truetype('/root/.config/Ultralytics/Arial.ttf', 20)
        txt_w, txt_h = 40, 30  # 94, 35
        txt_outside = box[0] - txt_h >= 0  # label fits outside box
        img_cls_2_draw.rectangle([box[0],
                            box[1] - txt_h if txt_outside else box[1],
                            box[0] + txt_w + 1,
                            box[1] + 1 if txt_outside else box[1] + txt_h + 1], fill=(0, 0, 0))
        img_cls_2_draw.text((box[0], box[1] - txt_h if txt_outside else box[1]), 'chip', fill=(255, 255, 255), font=txt_font)
        img_cls_2_draw.rectangle(box, fill=None, outline=(255, 255, 255), width=2)
        img_cls_2_draw.point(((box[0] + box[2])/2, (box[1] + box[3])/2), fill=(255, 0, 0))
        img_cls_2_draw.chord([box[0]-5, box[1]-5, box[0]+5, box[1]+5], 0, 360, fill=(255, 0, 0))

    # _gt_bboxes = l[_ind_cls1_l, 1:]  # normalized xywh
    _gt_bboxes = l[:, 1:]
    _gt_bboxes_area = _gt_bboxes[:, 2] * _gt_bboxes[:, 3]
    _sm_obj_mask = _gt_bboxes_area < 0.01  # 中小目标的面积阈值, 根据数据集及应用场景而设定, 高于该值的目标无需crop放大;
    _gt_bboxes = bbox_cxcywh_to_xyxy(torch.from_numpy(_gt_bboxes[_sm_obj_mask]))
    cluster_label = cluster_gt_bboxes(_gt_bboxes)
    _enclosed_ltrb = torch.cat((_gt_bboxes[:, :2].min(dim=0)[0], _gt_bboxes[:, 2:].max(dim=0)[0]),-1)
    _enclosed_ltrb[[0, 2]] *= img_wh[0]
    _enclosed_ltrb[[1, 3]] *= img_wh[1]
    _cluster_color = [(0,255,0), (0,0,255), (255,255,0)]  # lime, blue, yellow
    for _cluster_id in [1, 2, 3]:
        _ind_cluster = torch.nonzero(cluster_label[:, 0] == _cluster_id, as_tuple=False).squeeze()
        if _ind_cluster.numel() < 1:
            continue
        elif _ind_cluster.numel() == 1:
            _chip_ltrb = _gt_bboxes[_ind_cluster, :].clone()
            _ind_cluster = [_ind_cluster, ]
        else:
            _chip_ltrb = torch.cat((_gt_bboxes[_ind_cluster, :2].min(dim=0)[0], _gt_bboxes[_ind_cluster, 2:].max(dim=0)[0]),-1)
        _chip_ltrb[[0, 2]] *= img_wh[0]
        _chip_ltrb[[1, 3]] *= img_wh[1]
        img_cls_2_draw.rectangle(_chip_ltrb.cpu().numpy(), fill=None, outline=_cluster_color[_cluster_id-1], width=2)
        for _ind in _ind_cluster:
            _bbox = _gt_bboxes[_ind].cpu().numpy()
            _bbox_ctr_x = (_bbox[0] + _bbox[2])/2 * img_wh[0]
            _bbox_ctr_y = (_bbox[1] + _bbox[3])/2 * img_wh[1]
            img_cls_2_draw.chord([_bbox_ctr_x-6, _bbox_ctr_y-6, _bbox_ctr_x+6, _bbox_ctr_y+6], 0, 360, fill=_cluster_color[_cluster_id-1])
            if cluster_label[_ind, 1] == 1:
                img_cls_2_draw.chord([_bbox_ctr_x-3, _bbox_ctr_y-3, _bbox_ctr_x+3, _bbox_ctr_y+3], 0, 360, fill=(255,0,0))

    img_cls2.save(path_to_tmp / f'img_cls_filter_l_cluster_v1_diouv_3cluster_1.2_3.2.jpg')
    pass


if __name__ == "__main__":
    main()
    print('\nfinish')
