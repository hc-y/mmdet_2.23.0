#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file    : plot_cdf_dataset.py
@path_to_file:
@date    : 2022/06/07 16:13
@contact :
@brief   :  计算每张图片中 objects 相对于图片大小的相对尺度, 绘制整个数据集中不同相对尺度大小的 objects 比例分布;
@intro   :
@relatedfile:
    https://www.geeksforgeeks.org/how-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python/

@annotation: hc-y_note:, hc-y_Q:, hc-y_highlight:, hc-y_add:, hc-y_modify:, c-y_write:,
"""
import glob
import sys
from pathlib import Path

import json
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
from mmdet.utils import increment_path
from mmdet.utils import plot_images_v1


def plot_cdf_scale(json_dir):
    if False:
        t0 = time.time()
        with open(json_dir, 'r') as f:
            a = json.load(f)
        area_ratio_total = []
        for _img in a['images']:
            areas_per_img = np.array([_ann['bbox'][-2] * _ann['bbox'][-1] for _ann in a['annotations'] if _ann['image_id'] == _img['id']], dtype=np.float64)
            img_area = _img['width'] * _img['height']
            # rel_scale = [pow(ann['bbox'][-2] * ann['bbox'][-1] / img_area, .5) for ann in anns_per_img]
            # rel_scale2 = [pow(ann['area'] / img_area, .5) for ann in anns_per_img]
            area_ratio_total.append(areas_per_img / img_area)
        area_ratio_total = np.concatenate(area_ratio_total)
        np.save('./my_workspace/tmp/area_ratio_argoverse.npy', area_ratio_total)  # area_ratio_coco; area_ratio_argoverse;
        print(f'cost {(time.time() - t0) / 60:.3f} minutes.')  # cost 440.489 minutes for coco; cost 51.858 minutes for argoverse-HD;
    rel_scale_total = np.sqrt(np.load('./my_workspace/tmp/area_ratio_coco.npy'))
    rel_scale_total2 = np.sqrt(np.load('./my_workspace/tmp/area_ratio_argoverse.npy'))
    
    fig = plt.figure(figsize=(12, 6), dpi=100)  # 生成图 (figure)
    ax = fig.add_subplot()
    color_list = ['dodgerblue', 'darkorange', 'green']
    line_label_list = ['Argoverse-HD', 'COCO']
    for i, data in enumerate([rel_scale_total2, rel_scale_total]):
        # # No of Data points
        # N = 500
        # # initializing random values
        # data = np.random.randn(N)
        # getting data of the histogram
        # bins_count, bins_edge = np.histogram(np.array(data), bins=100)
        bins_count, bins_edge = np.histogram(data, bins=np.linspace(0, 1, 101))
        # finding the PDF of the histogram using bins_count values
        pdf = bins_count / sum(bins_count)
        # using np.cumsum to calculate the CDF
        cdf = np.concatenate([np.array([0.]),np.cumsum(pdf)])
        # plotting PDF and CDF
        # ax.plot(bins_edge[1:], pdf, color="red", label="PDF")
        ax.plot(bins_edge, cdf, color=color_list[i], linewidth=2.0, label=line_label_list[i])
        rel_scale_given = [0.3, 0.5, 0.7]
        cdf_at = [np.interp(_val, cdf, bins_edge) for _val in rel_scale_given]
        ax.plot(cdf_at, rel_scale_given, 'ko')
        for _xy in zip(cdf_at, rel_scale_given):
            if i == 1:
                plt.text(_xy[0]+0.005, _xy[1]-0.04, f'({_xy[0]:.3f},{_xy[1]})')
            elif i == 0:
                plt.text(_xy[0]+0.005, _xy[1]+0.02, f'({_xy[0]:.3f},{_xy[1]})')
            ax.plot([_xy[0], _xy[0]], [0., _xy[1]], color=color_list[i], linewidth=1.0, linestyle=(0, (3, 3)), )
        cdf_given = [32, 96]
        rel_scale_at = [np.interp(pow(_val**2/(1920*1200), .5), bins_edge, cdf) for _val in cdf_given]
    # plt.xticks(np.linspace(-2, 3, 11))
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.axis([0., 1., 0., 1.])
    ax.grid()
    plt.xlabel('Relative Scale')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right', bbox_to_anchor=(0.99, 0.025))
    
    plt.tight_layout()
    plt.savefig("./my_workspace/tmp/cdf_coco_argoverse.png")


def main():
    # dir = Path('/home/hustget/hustget_workdir/yuhangcheng/Pytorch_WorkSpace/OpenSourcePlatform/datasets')
    dir = Path('/media/hustget/HUSTGET/amax/HUSTGET_users/yuhangcheng/OpenSourcePlatform/datasets')
    annotations_dir = 'Argoverse-1.1/annotations'
    # annotations_dir = 'Argoverse-HD-mini/annotations'
    str_train = 'train'
    # annotations_dir = 'coco/annotations'
    # str_train = 'instances_train2017'
    plot_cdf_scale(dir / annotations_dir / f"{str_train}.json")
    print('\nfinish!')


if __name__ == "__main__":
    main()
