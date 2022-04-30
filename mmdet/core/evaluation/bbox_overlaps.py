# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  is_aligned=False,  # hc-y_add0430:
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4) in <x1, y1, x2, y2> format or empty;
        bboxes2 (ndarray): Shape (k, 4) in <x1, y1, x2, y2> format or empty;
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        is_aligned (bool, optional): If True, then n and k must be equal.
            Default False.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k) if ``is_aligned`` is False else shape (n,)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols
        ious = np.zeros((rows,), dtype=np.float32)
    else:
        ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    if is_aligned:
        x_start = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[:, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1 + area2 - overlap
        else:
            union = area1 if not exchange else area2
        union = np.maximum(union, eps)
        ious = overlap / union
    else:
        for i in range(bboxes1.shape[0]):
            x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
            y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
            x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
            y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
            overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
                y_end - y_start + extra_length, 0)
            if mode == 'iou':
                union = area1[i] + area2 - overlap
            else:
                union = area1[i] if not exchange else area2
            union = np.maximum(union, eps)
            ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
