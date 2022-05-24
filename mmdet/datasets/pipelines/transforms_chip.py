# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import xyxy2xywh, xyxy2xywhn, xywh2xyxy, increment_path, clip_coords
from ..builder import PIPELINES
from .transforms import Resize, Pad, Normalize

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class RemixChipsV1v1:  # hc-y_add0419:
    """Remix images & bbox.

    This tranform crops several chips from the input high-resolution image, 
    and then rescales these chips and remixes them to produce a new image.
    """
    def __init__(self, tmp_repr=None):
        self.tmp_repr = tmp_repr
        pass

    def __call__(self, results):
        """Call function to resize images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        pass
        self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(tmp_repr={self.tmp_repr}, '
        return repr_str


@PIPELINES.register_module()
class ResizeChipsV1v1(Resize):  # hc-y_add0104:
    """hc-y_add0104:将原始图片均匀切分成4块并 resize 到指定大小;"""
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            # hc-y_add0104:需确保 img_sz 是32的整数倍, chip_sz 也是32的整数倍, 从而使得 img, chip 无需 Pad 就可以作为 model input;
            h0, w0 = results[key].shape[:2]
            chip4_list, labels4_list, chip3_fts_idx = [], [], []
            h0_half, w0_half = int(h0 / 2), int(w0 / 2)  # 注意: w0_half * 2 和 h0_half * 2 分别都有可能小于 w0和h0;
            for j in range(4):
                if j == 0:  # top left
                    x1a, y1a, x2a, y2a = 0, 0, w0_half, h0_half
                elif j == 1:  # top right
                    x1a, y1a, x2a, y2a = w0_half, 0, w0_half * 2, h0_half
                elif j == 2:  # bottom left
                    x1a, y1a, x2a, y2a = 0, h0_half, w0_half, h0_half * 2
                elif j == 3:  # bottom right
                    x1a, y1a, x2a, y2a = w0_half, h0_half, w0_half * 2, h0_half * 2
                    assert x2a == w and y2a == h
                chip4_list.append(cv2.resize(results[key][y1a:y2a, x1a:x2a], (1024, 640), interpolation=cv2.INTER_LINEAR))
            results['chip4'] = chip4_list
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class ResizeChipsV1v2(Resize):  # hc-y_add0105:
    """hc-y_add0105:crop chips with different FoV, rescale and append to list;"""
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            # hc-y_add0104:需确保 img_sz 是32的整数倍, chip_sz 也是32的整数倍, 从而使得 img, chip 无需 Pad 就可以作为 model input;
            img0 = results[key]
            gt_anns = np.concatenate((results['gt_labels'][:,None], results['gt_bboxes']), 1) if len(results.get('gt_labels', [])) else None # (category,x1,y1,x2,y2)
            hw_unpad, hw_pad, pad_ltrb = self.letterbox_pad_params(img.shape[:2], divisor=32)
            # self.calculate_chip_xywh_after(results[key].shape[:2], hw_unpad, pad_ltrb, (h_scale, w_scale), ftsfus_stride_max=32)  # hc-y_add0114:
            ftsfus_stride_max = 32
            chips_xywh = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
            offsets_topleft = np.array([[1, 0], [1, 0], [1, 0]])
            # append each chip to chip4_list; update labels in each chip to labels4_list;
            chip4_list, labels4_list, chip3_fts_idx = [], [], []
            flag_wo_ftsftu = False
            if flag_wo_ftsftu:
                chip4_list.append(img)
                chip3_fts_idx.append(None)
            else:
                # from mmdet.utils import increment_path, xyxy2xywhn
                # path_to_tmp = increment_path('/mnt/data2/yuhangcheng/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server58
                # path_to_tmp = increment_path('/mnt/data1/yuhangcheng/yhc_workspace/mmdet_1213/my_workspace/tmp/', exist_ok=False, mkdir=True)  # server57
                # cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
                # from mmdet.utils import plot_images_v1
                for j in range(4):
                    labels = None if gt_anns is None else gt_anns.copy()
                    if j == 0:
                        chip = img
                        if gt_anns is not None:
                            # hc-y_note0106:参考 def _resize_bboxes(self, results)
                            labels[:, 1:] = labels[:, 1:] * scale_factor
                            if self.bbox_clip_border:
                                labels[:, 1::2] = np.clip(labels[:, 1::2], 0, hw_unpad[1])
                                labels[:, 2::2] = np.clip(labels[:, 2::2], 0, hw_unpad[0])
                    elif j > 0:
                        # (x1aa,y1aa,waa,haa)表示chip在model input with the full-global FoV中的像素索引(left,top,width,height); 此处约束了chip在feature map of full-image中的特征索引为整数;
                        x1aa = math.floor((hw_unpad[1] * (chips_xywh[j-1,0] - chips_xywh[j-1,2] / 2) + pad_ltrb[0]) / ftsfus_stride_max + offsets_topleft[j-1,1]) * ftsfus_stride_max
                        y1aa = math.floor((hw_unpad[0] * (chips_xywh[j-1,1] - chips_xywh[j-1,3] / 2) + pad_ltrb[1]) / ftsfus_stride_max + offsets_topleft[j-1,0]) * ftsfus_stride_max
                        waa = math.floor(hw_unpad[1] * chips_xywh[j-1,2] / ftsfus_stride_max) * ftsfus_stride_max
                        haa = math.floor(hw_unpad[0] * chips_xywh[j-1,3] / ftsfus_stride_max) * ftsfus_stride_max

                        # (x1a0,y1a0,wa0,ha0)表示chip在原始高分辨率图片中的像素索引(left,top,width,height)
                        x1a0, wa0 = (int(_val / w_scale) for _val in (x1aa - pad_ltrb[0], waa))
                        y1a0, ha0 = (int(_val / h_scale) for _val in (y1aa - pad_ltrb[1], haa))

                        h0_chip_canvas, w0_chip_canvas = (math.floor(_val / ftsfus_stride_max) * ftsfus_stride_max for _val in hw_pad)
                        ratio_chip_canvas = min(w0_chip_canvas/wa0, h0_chip_canvas/ha0)
                        # (0,0,wbb,hbb)表示chip在model input with the local FoV中的像素索引
                        wbb, hbb = (round(_val * ratio_chip_canvas) for _val in (wa0, ha0))
                        assert wbb%ftsfus_stride_max == 0 or hbb%ftsfus_stride_max == 0
                        # hw_pad = (hbb, wbb)  # 注释此行时, chip4_list 中的各个 chip 通过 padding 而保证宽高一致;

                        chip3_fts_idx.append(np.array(((x1aa, y1aa, waa, haa), (0, 0, wbb, hbb))))

                        chip = np.full((hw_pad[0], hw_pad[1], img.shape[2]), 114, dtype=np.uint8)  # padded chip as one of the model inputs
                        # place the chip cropped from original image
                        chip[:hbb, :wbb] = cv2.resize(img0[y1a0:(y1a0+ha0), x1a0:(x1a0+wa0)], (wbb, hbb), interpolation=cv2.INTER_LINEAR)

                        if gt_anns is not None:
                            gt_bbox_wh_before_clip = np.stack((labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]), 1)
                            np.clip(labels[:, 1::2], x1a0, x1a0+wa0, out=labels[:, 1::2])
                            np.clip(labels[:, 2::2], y1a0, y1a0+ha0, out=labels[:, 2::2])
                            gt_bbox_wh_after_clip = np.stack((labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]), 1)
                            idx_del = np.where(np.amin(gt_bbox_wh_after_clip / gt_bbox_wh_before_clip, axis=1) < 1/3)[0]  # 当gt bbox被clip掉大部分比例时, 直接删除该gt bbox;
                            labels = np.delete(labels, idx_del, axis=0)
                            labels[:, 1::2] = labels[:, 1::2] - x1a0
                            labels[:, 2::2] = labels[:, 2::2] - y1a0
                            labels[:, 1:] = labels[:, 1:] * ratio_chip_canvas
                    # cv2.imwrite(str(path_to_tmp / f'img_chip{j}.jpg'), chip)  # save
                    # labels[:, 1:] = xyxy2xywhn(labels[:, 1:], hw_pad[1], hw_pad[0])
                    # plot_images_v1(None, np.concatenate((np.zeros_like(labels[:,0:1]), labels), -1), (str(path_to_tmp / f'img_chip{j}.jpg'), ), path_to_tmp / f'img_chip{j}_gts.jpg', cls_names, None, 'original_image')
                    # print('')

                    chip4_list.append(chip)
                    labels4_list.append(labels)
            results['chip4'] = chip4_list
            results['chip3_fts_idx'] = chip3_fts_idx
            results[key] = img

            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def letterbox_pad_params(self, shape_resized, divisor=32):
        """
        hc-y_note0105:注意确保 padding 方式 与 mmdet_1213/mmdet/datasets/pipelines/transforms.py def _pad_img() --> mmcv.impad_to_multiple() 一致;
        Args:
            shape_resized (Tuple): (height, width);
        Returns:
            shape_pad (Tuple): (height, width);
            padding_ltrb (Tuple): the padding for the left, top, right and bottom borders respectively;
        """
        pad_h = int(np.ceil(shape_resized[0] / divisor)) * divisor
        pad_w = int(np.ceil(shape_resized[1] / divisor)) * divisor
        shape_pad = (pad_h, pad_w)
        # left, top, right, bottom, 
        padding_ltrb = (0, 0, shape_pad[1] - shape_resized[1], shape_pad[0] - shape_resized[0])
        
        return shape_resized, shape_pad, padding_ltrb

    def calculate_chip_xywh_after(self, hw_img0, hw_unpad, pad_ltrb, hw_scale, ftsfus_stride_max=32):  # hc-y_add0114:
        # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1280, 800), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.52, 0.5, 0.48) <-- (0.5, 0.5, 0.5, 0.5)
        # hc-y_note0114:imgsz=(1920, 1200), inputsz=(1024, 640), offset_topleft = (1, 0)时, chip_xywh = (0.5, 0.55, 0.5, 0.5) <-- (0.5, 0.5, 0.5, 0.5)
        chips_xywh = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.8]])
        offsets_topleft = np.array([[1, 0], [1, 0], [1, 0]])
        h_scale, w_scale = hw_scale[0], hw_scale[1]
        for j in range(1, 4):
            x1aa = math.floor((hw_unpad[1] * (chips_xywh[j-1,0] - chips_xywh[j-1,2] / 2) + pad_ltrb[0]) / ftsfus_stride_max + offsets_topleft[j-1,1]) * ftsfus_stride_max
            y1aa = math.floor((hw_unpad[0] * (chips_xywh[j-1,1] - chips_xywh[j-1,3] / 2) + pad_ltrb[1]) / ftsfus_stride_max + offsets_topleft[j-1,0]) * ftsfus_stride_max
            waa = math.floor(hw_unpad[1] * chips_xywh[j-1,2] / ftsfus_stride_max) * ftsfus_stride_max
            haa = math.floor(hw_unpad[0] * chips_xywh[j-1,3] / ftsfus_stride_max) * ftsfus_stride_max

            x1a0, wa0 = (int(_val / w_scale) for _val in (x1aa - pad_ltrb[0], waa))
            y1a0, ha0 = (int(_val / h_scale) for _val in (y1aa - pad_ltrb[1], haa))
            _chip_xywh_2 = np.array([(x1a0 + wa0/2) / hw_img0[1], (y1a0 + ha0/2) / hw_img0[0], wa0 / hw_img0[1], ha0 / hw_img0[0]])
            print(f'imgsz={(hw_img0[1], hw_img0[0])}, inputsz={(hw_unpad[1], hw_unpad[0])}, offset_topleft={offsets_topleft[j-1]}时, chip_xywh = {_chip_xywh_2} <-- {chips_xywh[j-1]}')
            print(f'chip_xywh before round: {chips_xywh[j-1]}\nchip_xywh after round: {_chip_xywh_2}\n')


@PIPELINES.register_module()
class ResizeChipsV1v3(Resize):  # hc-y_add0501:
    """hc-y_add0501:cluster to generate chips based on det_result of last frame, 
        crop chips from original image of current frame, resize them and append to list;"""

    def _cluster_gt_bboxes_ndarray(self, _p_bboxes, img_wh, score_thr=0.3):
        """hc-y_write0503:基于 DensityPeakCluster 生成 small objects 的聚集区域;
        注:数据在 ndarray 和 Torch.Tensor 这两种形式下的有效数字位数会不一样, e.g., p_rho 的值会有差异;

        Args:
            _p_bboxes (ndarray): gt bboxes with shape (N, 5) in (x1, y1, x2, y2, score) format;
            p_bboxes (ndarray): gt bboxes with shape (N, 4) in (x1, y1, x2, y2) format;
            img_wh (Tuple): width and height of original image;
            score_thr (float): Minimum score of bboxes to be clustered. Default: 0.3.

        Returns:
            cluster_label (ndarray): with shape (N, 2), which cluster each gt bbox belongs.
                        一个用于指示划分到了哪个簇, 一个用于指示是否是簇中心;
            chips_ltrb (ndarray): with shape (<=3,4), 生成的原始 chips 参数;
            chips_ltrb_expand_new (v): with shape (<=3,4), 过滤掉被其它 chips_expand 所包含
                        的以及所包含 objects 数量少于3的 chips 后, 剩余 chips 所对应的 chips_expand 参数;
        """
        high_score_inds = np.where(_p_bboxes[:, 4] > score_thr)[0]  # 0.3, 0.2; 参考 mmdet/models/detectors/base.py def show_result(score_thr=0.3);
        _p_bboxes_xywh = xyxy2xywhn(_p_bboxes[high_score_inds, :4], w=img_wh[0], h=img_wh[1])
        _sm_obj_mask = _p_bboxes_xywh[:, 2] * _p_bboxes_xywh[:, 3] < 0.01  # 中小目标的面积阈值, 根据数据集及应用场景而设定, 高于该值的目标无需crop放大;
        p_bboxes = xywh2xyxy(_p_bboxes_xywh[_sm_obj_mask])
        # p_bboxes = p_bboxes.cpu().numpy()
        num_p = len(p_bboxes)
        if num_p <= 2:
            return None
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
            return None
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

        return chips_ltrb_expand_new  # cluster_label, chips_ltrb, chips_ltrb_expand_new


    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        det_result = results.pop('det_result')
        if len(det_result) > 0:
            # lf: last frame; cf: current frame; nf: next frame;
            chips_lf = self._cluster_gt_bboxes_ndarray(det_result[0]['det_bbox'], det_result[0]['img_shape'][:2][::-1], score_thr=0.3)
            if False:
            # if True:
                det_bbox, det_label = det_result[0]['det_bbox'], det_result[0]['det_label']
                from pathlib import Path
                path_to_tmp = Path('./my_workspace/tmp/')
                path_to_img = results['filename']
                img_wh = det_result[0]['img_shape'][:2][::-1]
                high_score_inds = np.where(det_bbox[:, 4] > 0.2)[0]  # 0.3, 0.2; mmdet/models/detectors/base.py def show_result(score_thr=0.3)
                _p_bboxes_xywh = xyxy2xywhn(det_bbox[high_score_inds, :4], w=img_wh[0], h=img_wh[1])
                _sm_obj_mask = _p_bboxes_xywh[:, 2] * _p_bboxes_xywh[:, 3] < 0.01  # 中小目标的面积阈值, 根据数据集及应用场景而设定, 高于该值的目标无需crop放大;
                bbox_vis = np.concatenate((_p_bboxes_xywh[_sm_obj_mask], det_bbox[high_score_inds[_sm_obj_mask], 4:5]), -1)
                label_vis = det_label[high_score_inds[_sm_obj_mask]]
                l_dets = np.concatenate((np.zeros_like(label_vis[:,None]), label_vis[:,None], bbox_vis), -1)
                from mmdet.utils import plot_images_v1
                cls_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck', 6: 'traffic_light', 7: 'stop_sign'}
                plot_images_v1(None, l_dets, (path_to_img, ), path_to_tmp / f'img_merged1_sm.jpg', cls_names, None, 'original_image')
                from PIL import Image
                from tools.focus_argoverse_dataset import DashedImageDraw
                img_src = Image.open(path_to_tmp / f'img_merged1.jpg')
                img_src_draw = DashedImageDraw(img_src)
                if chips_lf is not None:
                    for _chip_ltrb_expand_new in chips_lf:
                        img_src_draw.dashed_rectangle(_chip_ltrb_expand_new, dash=(8,8), outline=(255,255,255), width=3)
                img_src.save(path_to_tmp / f"img_merged1_cls_euc_3cluster.jpg")
            # chips_lf = np.array([[50.,50.,500.,500.],[100.,100.,800.,800.]])
        else:  # 对于 img_info['fid'] == 0 的 frame, 手动选取一个中心区域得到一个 chip;
            chip_xywh = (0.5, 0.5, 0.5, 0.5)
            chips_lf = xywh2xyxy(np.array([chip_xywh]))
            chips_lf[:, 0::2] *= results['ori_shape'][1]
            chips_lf[:, 1::2] *= results['ori_shape'][0]
            # chips_lf = None

        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

            chips_fields = []
            if chips_lf is not None:
                img0 = results[key]
                for i in range(len(chips_lf)):
                    x1a0,y1a0,x2a0,y2a0 = (int(_val) for _val in chips_lf[i])
                    chip_img0 = img0[y1a0:y2a0, x1a0:x2a0]
                    if self.keep_ratio:
                        chip_img, chip_scale_factor = mmcv.imrescale(
                            chip_img0,
                            results['scale'],
                            return_scale=True,
                            backend=self.backend)
                        chip_new_h, chip_new_w = chip_img.shape[:2]
                        chip_h, chip_w = chip_img0.shape[:2]
                        chip_w_scale = chip_new_w / chip_w
                        chip_h_scale = chip_new_h / chip_h
                    else:
                        chip_img, chip_w_scale, chip_h_scale = mmcv.imresize(
                            chip_img0,
                            results['scale'],
                            return_scale=True,
                            backend=self.backend)
                    chip_scale_factor = np.array(
                        [chip_w_scale, chip_h_scale, chip_w_scale, chip_h_scale], dtype=np.float32)
                    chip_fields = dict(cimg=chip_img, c_ltrb=(x1a0,y1a0,x2a0,y2a0), img_shape= \
                        chip_img.shape, pad_shape=chip_img.shape, scale_factor= chip_scale_factor)
                    chips_fields.append(chip_fields)

            results['chips_fields'] = chips_fields
            results[key] = img

            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class PadChipsV1v1(Pad):  # hc-y_add0502:
    """hc-y_add0502:Pad chips;"""

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
                for chip_fields in results.get('chips_fields', []):
                    chip_img_padded = mmcv.impad(chip_fields['cimg'], shape=self.size, pad_val=pad_val)
                    chip_fields['cimg'] = chip_img_padded
                    chip_fields['img_shape'] = chip_img_padded.shape
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
                for chip_fields in results.get('chips_fields', []):
                    raise NotImplementedError('hc-y_TODO.')
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


@PIPELINES.register_module()
class NormalizeChipsV1v1(Normalize):
    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        if 'chip4' in results:
            for j in range(len(results['chip4'])):
                results['chip4'][j] = mmcv.imnormalize(results['chip4'][j], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results
