#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file    : split_argoverse_dataset.py
@path_to_file:
@date    : 2022/01/18 14:30
@contact :
@brief   : 依据['all', 's_inner', 'ml_inner', 's_outer', 'ml_outer']拆分Argoverse-HD的annotations;
@intro   :
@relatedfile:
    

@annotation: hc-y_note:, hc-y_Q:, hc-y_highlight:, hc-y_add:, hc-y_modify:, c-y_write:,
"""
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
from mmdet.utils import xyxy2xywhn, xywh2xyxy, increment_path
from mmdet.utils import plot_images_v1


def split_gts_scale_type(json_dir):  # hc-y_add0118:
    scale_type = ['all', 's_inner', 'ml_inner', 's_outer', 'ml_outer']
    # scale_type = ['all', 's', 'ml']
    # scale_type = ['all', 'inner', 'outer']
    num_scale_type = len(scale_type) - 1
    with open(json_dir, 'r') as f:
        a = json.load(f)
    a_sub = [dict() for _ in range(num_scale_type)]
    for k,v in a.items():
        if k == 'images':
            for idx_type in range(num_scale_type):
                a_sub[idx_type][k] = v
            img_id_hw_list = [(_img['id'], _img['height'], _img['width']) for _img in v]
        elif k == 'annotations':
            anns_list = [[] for _ in range(num_scale_type)]
            for _img_id_hw in img_id_hw_list:
                anns_per_img = [_ann for _ann in v if _ann['image_id'] == _img_id_hw[0]]
                if len(anns_per_img) == 0:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                _gt_bboxes = np.array([_val['bbox'] for _val in anns_per_img], dtype=np.float64)
                _gt_bboxes[:, :2] += _gt_bboxes[:, 2:] / 2  # xy top-left corner to center
                _gt_bboxes[:, [0, 2]] /= _img_id_hw[2]  # normalize x
                _gt_bboxes[:, [1, 3]] /= _img_id_hw[1]  # normalize y
                _gt_idx_split = split_bboxes_by_scale(_gt_bboxes, split_type=scale_type)
                for idx_type in range(num_scale_type):
                    for _gt_idx in _gt_idx_split[idx_type+1]:
                        anns_list[idx_type].append(anns_per_img[_gt_idx])
            
            for idx_type in range(num_scale_type):
                a_sub[idx_type][k] = anns_list[idx_type]
        else:
            for idx_type in range(num_scale_type):
                a_sub[idx_type][k] = v
    
    path_to_new_file = json_dir.parents[1] / 'annotations_scale_type_800_split_by_iof'
    if not path_to_new_file.exists():
        path_to_new_file.mkdir(parents=True, exist_ok=True)
    for idx_type in range(num_scale_type):
        json.dump(a_sub[idx_type], open(path_to_new_file  / (json_dir.stem + f'_{scale_type[idx_type+1]}.json'), 'w'), indent=4)
    if "test-meta.json" in (json_dir.stem + '.json'):
        return


def split_bboxes_by_scale(boxes, split_type=None):
    """
    hc-y_note0117:区分"是属于small objects,还是属于medium,large objects", 区分"是位于chip内,还是位于chip外"; s,ml,in,out;
    Arguments:
        boxes: (Array[N, 4]), ctrx, ctry, w, h in normalized format;
        img_hw: (tuple), hw of original img
        bbox_type: (str), {'pred', 'label'}
    Returns:
        bboxes_idx_list: 
    """
    thr_scale_s = 0.05  # 超参数待设置; 0.05; 0.02;
    thr_inter_len = 1/3  # 超参数待设置;
    # 预定义1个chip用于划分inner/outer objects
    # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1280, 800), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.52, 0.5, 0.48) <-- (0.5, 0.5, 0.5, 0.5)
    chip_xywh = np.array([0.5, 0.52, 0.5, 0.48])
    # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1024, 640), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.55, 0.5, 0.5) <-- (0.5, 0.5, 0.5, 0.5)
    # chip_xywh = np.array([0.5, 0.55, 0.5, 0.5])
    chip_xyxy = xywh2xyxy(chip_xywh[None, :])[0]
    
    # 依据尺度(面积)大小划分small/medium/large objects;
    boxes_all_idx = np.arange(0, boxes.shape[0])
    boxes_s_idx_mask = boxes[:, 2] * boxes[:, 3] <= thr_scale_s ** 2
    boxes_ml_idx_mask = ~boxes_s_idx_mask

    _boxes_split_idx_list = []  # ['s_inner', 's_outer', 'ml_inner', 'ml_outer']:
    for _boxes_idx_mask in [boxes_s_idx_mask, boxes_ml_idx_mask]:
        _boxes_idx = boxes_all_idx[_boxes_idx_mask]
        if len(_boxes_idx) == 0:
            _boxes_split_idx_list.append(np.array([], dtype=np.int64))
            _boxes_split_idx_list.append(np.array([], dtype=np.int64))
        else:
            _boxes_xywh = boxes[_boxes_idx_mask]
            _boxes_xyxy = xywh2xyxy(_boxes_xywh)  # xyxy, (num_scale,4)
            inter_w = np.minimum(chip_xyxy[2], _boxes_xyxy[:, 2]) - np.maximum(chip_xyxy[0], _boxes_xyxy[:, 0])  # (num_scale,)
            inter_h = np.minimum(chip_xyxy[3], _boxes_xyxy[:, 3]) - np.maximum(chip_xyxy[1], _boxes_xyxy[:, 1])
            flag_split_type = 'split_by_iof'
            if flag_split_type == 'split_by_wh':
                inter_wh = np.concatenate([inter_w[:, None], inter_h[:, None]], 1)  # (num_scale,2)
                # 与chip有重叠且重叠区域的宽或高都大于其自身宽或高的几分之一的bboxes 判定为inner;  # hc-y_modify1126:
                _boxes_overlap_mask = ((inter_wh[..., 0] / _boxes_xywh[:,2]) >= thr_inter_len) * ((inter_wh[..., 1] / _boxes_xywh[:,3]) >= thr_inter_len)  # (num_scale,)
            elif flag_split_type == 'split_by_iof':
                inter_area = np.clip(inter_w, 0, 1) * np.clip(inter_h, 0, 1)  # (num_scale,)
                _boxes_area = _boxes_xywh[:, 2] * _boxes_xywh[:, 3]
                # 与chip有重叠且重叠区域的面积占其自身面积的比例大于的bboxes 判定为inner;  # hc-y_modify0117:
                _boxes_overlap_mask = inter_area / _boxes_area >= 0.6  # (num_scale,)
            else:
                pass
            _boxes_split_idx_list.append(_boxes_idx[_boxes_overlap_mask])
            _boxes_split_idx_list.append(_boxes_idx[~_boxes_overlap_mask])

    boxes_split_idx_list = [boxes_all_idx, _boxes_split_idx_list[0], _boxes_split_idx_list[2], _boxes_split_idx_list[1], _boxes_split_idx_list[3]]
    bboxes_idx_list = [boxes_all_idx, ]
    if split_type == ['all', 's_inner', 'ml_inner', 's_outer', 'ml_outer']:
        bboxes_idx_list = boxes_split_idx_list
    elif split_type == ['all', 's', 'ml']:
        bboxes_idx_list.append(np.concatenate(boxes_split_idx_list[1::2], dim=0))
        bboxes_idx_list.append(np.concatenate(boxes_split_idx_list[2::2], dim=0))
    elif split_type == ['all', 'inner', 'outer']:
        bboxes_idx_list.append(np.concatenate(boxes_split_idx_list[1:3], dim=0))
        bboxes_idx_list.append(np.concatenate(boxes_split_idx_list[3:5], dim=0))
    return bboxes_idx_list


def plot_labels_on_img(json_dir):
    imgs_dir = '/mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-1.1/images'
    json_files = sorted(glob.glob(str(json_dir), recursive=False))
    for _json_file in json_files:
        path_to_tmp = increment_path(str(Path(_json_file).parent / Path(_json_file).stem), exist_ok=False, mkdir=True)  # server57
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
            _img_file = imgs_dir + '/' + a['seq_dirs'][_img['sid']] + '/' + _img['name']
            _img_with_gts_name = _img['name'].split('.')[0] + '_gts.jpg'
            plot_images_v1(None, np.concatenate((np.zeros_like(_labels[:,0:1]), _labels), -1), (_img_file, ), path_to_tmp / _img_with_gts_name, cls_names, None, 'original_image')


def main():
    dir = Path('/mnt/data1/yuhangcheng/yhc_workspace/datasets')
    annotations_dir = 'Argoverse-1.1/annotations/'
    # annotations_dir = 'Argoverse-HD-mini/annotations/'
    split_gts_scale_type(dir / annotations_dir / "val.json")
    # plot_labels_on_img(dir / 'Argoverse-HD-mini/annotations/val*.json')
    # plot_labels_on_img(dir / 'Argoverse-HD-mini/annotations_scale_type_800_sub/val*.json')
    print('\nfinish!')


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    main()
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD
    # cd /mnt/data1/yuhangcheng/yhc_workspace/datasets/Argoverse-HD-mini
    # rm -rf *.json
